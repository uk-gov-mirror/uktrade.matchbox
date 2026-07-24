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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Self
from weakref import WeakSet

from platformdirs import user_cache_path

from matchlab import lineage
from matchlab.adapters import Adapter, DuckDBAdapter, Fingerprint
from matchlab.core.hash import HASH_FUNC

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

    def __init__(self, name: str, upstream: tuple[Step, ...] = ()) -> None:
        """Initialise a plan node with its name and direct inputs."""
        self.name = name
        self.upstream = tuple(upstream)
        self._fp: Fingerprint | None = None
        self._adapter: Adapter | None = None
        _LIVE_STEPS.add(self)

    def __repr__(self) -> str:
        """Return a short representation showing kind, name and collection state."""
        state = "collected" if self.is_collected else "lazy"
        return f"<{type(self).__name__} {self.name!r} {state}>"

    @property
    def is_collected(self) -> bool:
        """Whether this step has been materialised."""
        return self._fp is not None

    # -- plan identity ----------------------------------------------------------------

    @abstractmethod
    def _config_key(self) -> bytes:
        """Bytes identifying this step's configuration.

        For a source this includes a content hash of the data read, so the fingerprint
        tracks the warehouse. For every other kind it is pure configuration.
        """
        ...

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
          miss**, re-running the step and everything below it. Renaming an upstream
          step does this: identity already arrives via the parent fingerprint, but
          `Model._config_key` also records the input's name.

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
                    f"Input '{parent.name}' of '{self.name}' has no fingerprint yet."
                )
            parts.append(parent._fp)
        return HASH_FUNC(b"|".join(parts)).digest()

    # -- execution --------------------------------------------------------------------

    @abstractmethod
    def _execute(self, adapter: Adapter, fp: Fingerprint) -> None:
        """Compute this step and persist its artifact under `fp`."""
        ...

    def _ensure(self, adapter: Adapter) -> None:
        """Materialise this step unless its artifact is already stored."""
        self._adapter = adapter

        if self._fp is not None and (not self.stores or adapter.has(self._fp)):
            return

        fp = self._fingerprint()

        if not self.stores:
            self._fp = fp
            return

        if adapter.has(fp):  # cache hit — skip the work entirely
            self._fp = fp
            return

        self._execute(adapter, fp)
        self._fp = fp

    # -- public API -------------------------------------------------------------------

    def collect(self, adapter: Adapter | None = None) -> Self:
        """Materialise this step and everything it depends on.

        Steps whose artifact is already stored are skipped without being run, so
        re-collecting after adding a downstream step only does the new work.

        Args:
            adapter: Where to read and write artifacts. Defaults to the module-level
                adapter (a DuckDB store in the user cache directory).

        Returns:
            This step, now collected.
        """
        adapter = adapter or default_adapter()
        lineage.validate(self)
        for step in lineage.walk(self):
            step._ensure(adapter)
        return self

    def lineage(self) -> list[Step]:
        """Return this step and all its inputs, upstream-first."""
        return lineage.walk(self)

    def get_step(self, name: str) -> Step:
        """Return an input step by name, searching only this step's lineage."""
        return lineage.find(self, name)

    def draw(self) -> str:
        """Render this step's sub-plan as a tree."""
        return lineage.draw(self)

    # -- helpers for subclasses -------------------------------------------------------

    def _require_adapter(self) -> Adapter:
        """Return the adapter this step was collected into."""
        if self._adapter is None or self._fp is None:
            raise RuntimeError(
                f"Step '{self.name}' has not been collected. Call collect() first."
            )
        return self._adapter

    def _frame(self, adapter: Adapter) -> pl.DataFrame:
        """Return this step's data. Overridden by kinds that feed other steps."""
        raise NotImplementedError(
            f"{type(self).__name__} does not produce a queryable frame."
        )

    def _describe(self) -> dict[str, Any]:
        """Serialisable description of this step, for config hashing and plan dumps."""
        return {"kind": self.kind, "name": self.name}
