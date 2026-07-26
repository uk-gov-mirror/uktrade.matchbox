"""The lazy plan node.

A `Step` holds references to its **inputs only** (`upstream`). There is no registry,
no parent pointer, and no downstream list — "the DAG" is whatever is reachable
upstream from the node you hold, and lineage operations are pure functions of a root
node (`matchlab.lineage`).

Nothing is computed until `collect()`. Collection walks the plan upstream-first and
runs only the steps whose artifact is not already stored.

**Fingerprints identify artifacts.** A step's fingerprint combines its kind, its
configuration, and its inputs' fingerprints — so for everything downstream of a
source it is derivable from the *plan alone*, before any work happens, and `collect`
can skip a cached step without running it. Sources are the exception: raw data enters
there, so a source's configuration key includes a content hash of the data it read.
Constructing a fresh `Source` therefore re-reads the warehouse (the documented way to
refresh), while an existing `Source` object memoises its read.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Self
from weakref import WeakSet

from platformdirs import user_cache_path
from pydantic import BaseModel

from matchlab import lineage
from matchlab.adapters import Adapter, DuckDBAdapter, Fingerprint
from matchlab.core.hash import HASH_FUNC
from matchlab.lineage import StepStatus
from matchlab.progress import report

if TYPE_CHECKING:
    import polars as pl

CACHE_DIR = user_cache_path("matchlab")

# Every live step, weakly referenced. Weak refs do not keep steps alive, so Python
# reachability drives lifetime — which is what makes `gc()` able to reclaim the
# storage of plans you have dropped. A strong registry (the old `DAG.nodes`) made
# that impossible.
_LIVE_STEPS: WeakSet[Step] = WeakSet()

_DEFAULT_ADAPTER: Adapter | None = None


def set_default_adapter(adapter: Adapter | None) -> None:
    """Set the adapter used by `collect()` when none is passed. `None` resets it."""
    global _DEFAULT_ADAPTER  # noqa: PLW0603 - module-level default, by design
    _DEFAULT_ADAPTER = adapter


def default_adapter() -> Adapter:
    """Return the default adapter, creating a DuckDB store in the cache dir if unset."""
    global _DEFAULT_ADAPTER  # noqa: PLW0603 - module-level default, by design
    if _DEFAULT_ADAPTER is None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _DEFAULT_ADAPTER = DuckDBAdapter(CACHE_DIR / "store.duckdb")
    return _DEFAULT_ADAPTER


def gc(adapter: Adapter | None = None) -> int:
    """Drop stored artifacts belonging to no live step.

    Mark-and-sweep against the live set — **not** a per-step finalizer. Because
    artifacts are content-addressed, two distinct steps can share a fingerprint, so
    deleting one step's artifact when it is dropped would destroy a live sibling's
    data.

    Returns:
        The number of artifacts removed.
    """
    adapter = adapter or default_adapter()
    live = {step._fp for step in _LIVE_STEPS if step._fp is not None}
    return adapter.gc(live)


class Step(ABC):
    """A node in a lazy plan."""

    kind: ClassVar[str]
    #: Whether this kind of step persists an artifact. `Clean` is fused (False) unless
    #: it is collected directly.
    stores: ClassVar[bool] = True

    def __init__(self, upstream: tuple[Step, ...] = ()) -> None:
        """Initialise a plan node with its direct inputs.

        Steps have no names. They are identified by **position** — where they fall in
        `lineage.walk`, which is the order `collect` runs them in and the order
        `PlanDocument` lists them in. So `step 7` in a log, `[7]` in `draw()`, and
        `steps[7]` in a document are the same node.

        A position is not stored here, because it is not a property of the step: it
        belongs to the walk it came from, and the same step numbers differently in
        `walk(deduped)` and `walk(companies)`. Whoever does the walking passes it to
        whoever needs it — `collect` to its reporter, `draw` to its own renderer.

        Finding a result later is a separate matter, and a separate act:
        `Resolver.publish` points a **label** at a resolution. `Source` is the one step
        with a name, and it means something else again — a source's name is part of its
        output, prefixing every column it contributes and tagging its rows.
        """
        self.upstream = tuple(upstream)
        self._fp: Fingerprint | None = None
        self._adapter: Adapter | None = None
        _LIVE_STEPS.add(self)

    def __repr__(self) -> str:
        """Return a short representation showing kind and collection state."""
        return f"<{type(self).__name__} {'collected' if self.is_collected else 'lazy'}>"

    def __str__(self) -> str:
        """How this step appears in a drawing. `Source` adds its name."""
        return self.kind

    @property
    def is_collected(self) -> bool:
        """Whether this step has been materialised."""
        return self._fp is not None

    # -- plan identity ----------------------------------------------------------------

    @property
    @abstractmethod
    def config(self) -> BaseModel:
        """This step's settings, as a serialisable model.

        One model per step kind, in `matchlab.core.config`. It must carry everything
        this step's output depends on and nothing else — that is the invariant
        `_config_key` rests on, and the one to check when adding a setting. Omit
        something that changes the output and collect will hand back a stale artifact
        without re-running (see `_fingerprint`).

        Configs describe a step's own settings, not its inputs'. Edges live on
        `upstream`, and `_fingerprint` already folds in the parents' fingerprints.
        """
        ...

    def _config_key(self) -> bytes:
        """Bytes identifying this step's configuration.

        `Source` extends this with a content hash of the data it read, so that the
        fingerprint tracks the warehouse. For every other kind the config is the
        whole story.
        """
        return json.dumps(self.config.model_dump(mode="json"), sort_keys=True).encode()

    def _fingerprint(self) -> Fingerprint:
        """Address this step by kind, configuration and inputs.

        The key is derived from the *plan*, not from the step's output. That is what
        makes it computable before the step runs, which is what lets `_ensure` skip
        work: an output digest would only be knowable once the work was already done.
        Sources are the exception — they fold a hash of the data they read into
        `_config_key`, because no configuration reveals that the warehouse moved.

        The trade-off is that the key can disagree with the bytes in both directions:

        * config omits something that changes the output — a **stale hit**, because
          `_ensure` never runs the step and reads the old artifact. `SplinkLinker`
          without an explicit seed is a live example. The only defence is that every
          `_config_key` covers everything its step's output depends on; treat that as
          the invariant to protect when adding a step or a setting.
        * config includes something that doesn't change the output — a **spurious
          miss**, re-running the step and everything below it. No config does this
          today, and the way to keep it that way is to record settings only: a config
          that described an *input* would, since identity already arrives via the
          parent fingerprint. A setting that must point at an input points at its
          position, which is not redundant — it decides which input the setting
          applies to.

        There is also no early cutoff. A `Clean` whose SQL is reformatted but
        semantically unchanged invalidates the whole subtree beneath it.

        TODO(fingerprints): split this into an action key (plan-derived, as now)
        mapping to an output digest (content-derived, recorded by the adapter on
        write), and build a child's key from its parents' output digests rather than
        their action keys. That buys early cutoff, storage dedup and rename tolerance.
        It does **not** fix stale hits — an output digest governs propagation, never
        admission, so a step whose action key is a hit still never runs and its digest
        is never consulted. Costs an indirection table in the adapter contract, a hash
        of every artifact on write, and the loss of a work-set knowable before running.
        See PLAN.md, "Known limitations", for the full ledger.
        """
        parts: list[bytes] = [self.kind.encode(), self._config_key()]
        for parent in self.upstream:
            if parent._fp is None:  # pragma: no cover - collect orders upstream first
                raise RuntimeError(
                    f"A {parent.kind} input of this {self.kind} has no fingerprint yet."
                )
            parts.append(parent._fp)
        return HASH_FUNC(b"|".join(parts)).digest()

    # -- execution --------------------------------------------------------------------

    @abstractmethod
    def _execute(self, adapter: Adapter, fp: Fingerprint) -> None:
        """Compute this step and persist its artifact under `fp`."""
        ...

    def _ensure(self, adapter: Adapter) -> StepStatus:
        """Materialise this step unless its artifact is already stored.

        This classifies the outcome but reports none of it. `collect` holds the walk,
        and therefore each step's position — the thing a log line has to quote to be
        findable in the plan it printed — so reporting belongs there, with the single
        `matchlab.progress.Progress` that owns both channels. That is also what keeps
        the live tree and the log from ever disagreeing. This keeps to the work.

        Returns:
            What it took: `DONE` if this call computed the step, `CACHED` if the
            artifact was already stored, `FUSED` if this kind stores nothing.
        """
        self._adapter = adapter

        if self._fp is not None and (not self.stores or adapter.has(self._fp)):
            return StepStatus.FUSED if not self.stores else StepStatus.CACHED

        fp = self._fingerprint()

        if not self.stores:
            self._fp = fp
            return StepStatus.FUSED

        if adapter.has(fp):  # cache hit — skip the work entirely
            self._fp = fp
            return StepStatus.CACHED

        self._execute(adapter, fp)
        self._fp = fp
        return StepStatus.DONE

    # -- public API -------------------------------------------------------------------

    def collect(
        self, adapter: Adapter | None = None, progress: bool | None = None
    ) -> Self:
        """Materialise this step and everything it depends on.

        Steps whose artifact is already stored are skipped without being run, so
        re-collecting after adding a downstream step only does the new work.

        Args:
            adapter: Where to read and write artifacts. Defaults to the module-level
                adapter (a DuckDB store in the user cache directory).
            progress: Whether to draw the plan as a live tree, redrawn in place. `None`
                — the default — draws it at a terminal and logs the same tree plus a
                record per step anywhere else. See `matchlab.progress`.

        Returns:
            This step, now collected.
        """
        adapter = adapter or default_adapter()
        # One walk, used for both. It fixes each step's position, so a `step 7` the
        # reporter logs is the node it drew as `[7]`.
        steps = lineage.walk(self)
        with report(self, steps, progress) as reporter:
            for step in steps:
                reporter.begin(step)
                try:
                    status = step._ensure(adapter)
                except Exception:
                    reporter.end(step, StepStatus.FAILED)
                    raise
                reporter.end(step, status)
        return self

    def lineage(self) -> list[Step]:
        """Return this step and all its inputs, upstream-first."""
        return lineage.walk(self)

    def draw(self) -> str:
        """Render this step's sub-plan as a tree."""
        return lineage.draw(self)

    # -- helpers for subclasses -------------------------------------------------------

    def _collected(self) -> tuple[Adapter, Fingerprint]:
        """Return the adapter this step was collected into, and its fingerprint.

        Both together, because everything that reads a stored artifact needs both and
        neither exists before collection — returning the pair is what lets a caller
        use the fingerprint without re-checking that it is there.
        """
        if self._adapter is None or self._fp is None:
            raise RuntimeError(
                f"This {self.kind} has not been collected. Call collect() first."
            )
        return self._adapter, self._fp

    def _require_adapter(self) -> Adapter:
        """Return the adapter this step was collected into."""
        return self._collected()[0]

    def _frame(self, adapter: Adapter) -> pl.DataFrame:
        """Return this step's data. Overridden by kinds that feed other steps."""
        raise NotImplementedError(
            f"{type(self).__name__} does not produce a queryable frame."
        )
