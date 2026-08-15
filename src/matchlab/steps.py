"""The lazy plan node.

A `Step` holds references to its **inputs only** (`parents`). There is no registry,
no parent pointer, and no downstream list. "The DAG" is whatever is reachable
upstream from the node you hold, and lineage operations are pure functions of a root
node (`matchlab.lineage`).

Nothing is computed until `collect()`. Collection walks the plan upstream-first and
runs only the steps whose artifact is not already stored.

**Fingerprints identify artifacts.** A step's fingerprint combines its kind, its
spec, and its inputs' fingerprints. Everything downstream of a source can therefore
derive it from the *plan alone*, before any work happens, and `collect`
can skip a cached step without running it. Sources are the exception. Raw data enters
there, so a source's spec key includes a content hash of the data it read.
Constructing a fresh `Source` therefore re-reads the warehouse (the documented way to
refresh), while an existing `Source` object memoises its read.

**A step is settled once built.** Everything its spec is derived from refuses
assignment, so a plan is changed by rebuilding it, not by editing a node in place.
"""

import json
from abc import ABC, abstractmethod
from typing import ClassVar, Self

from platformdirs import user_cache_path
from pydantic import BaseModel

from matchlab import lineage
from matchlab.adapters import Adapter, DuckDBAdapter, Fingerprint
from matchlab.core.hash import HASH_FUNC
from matchlab.core.kinds import StepKind
from matchlab.lineage import StepStatus
from matchlab.progress import report

CACHE_DIR = user_cache_path("matchlab")

_DEFAULT_ADAPTER: Adapter | None = None


def set_default_adapter(adapter: Adapter | None) -> None:
    """Set the adapter used by `collect()` when none is passed. `None` resets it."""
    global _DEFAULT_ADAPTER  # module-level default, by design
    _DEFAULT_ADAPTER = adapter


def default_adapter() -> Adapter:
    """Return the default adapter, creating a DuckDB store in the cache dir if unset."""
    global _DEFAULT_ADAPTER  # module-level default, by design
    if _DEFAULT_ADAPTER is None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _DEFAULT_ADAPTER = DuckDBAdapter(CACHE_DIR / "store.duckdb")
    return _DEFAULT_ADAPTER


class Step(ABC):
    """A node in a lazy plan."""

    kind: ClassVar[StepKind]

    @property
    @abstractmethod
    def parents(self) -> "tuple[Step, ...]":
        """This step's direct inputs, in the order they fold into its fingerprint.

        The kind-agnostic view of a step's edges.
        """
        ...

    def __init__(self) -> None:
        """Initialise a plan node.

        Steps have no names. They are identified by **position**, where they fall in
        `lineage.walk`, which is the order `collect` runs them in and the order
        `PlanDocument` lists them in. `step 7` in a log, `[7]` in `draw()`, and
        `steps[7]` in a document are therefore the same node.

        A position is not stored here, because it is not a property of the step. It
        belongs to the walk it came from, and the same step numbers differently in
        `walk(deduped)` and `walk(companies)`. Whoever does the walking passes it to
        whoever needs it, `collect` to its reporter, `draw` to its own renderer.

        Finding a result later is a separate matter, and a separate act.
        `Resolver.publish` points a **label** at a resolver's output. `Source` is the
        one step with a name, and it means something else again. A source's name is
        part of its output, prefixing every column it contributes and tagging its rows.
        """
        self._fp: Fingerprint | None = None
        self._adapter: Adapter | None = None

    # -- immutability -----------------------------------------------------------------
    #
    # A step builds what it runs from its settings at construction, so an assignment
    # afterwards would move one attribute and nothing derived from it: setting
    # `model_class` leaves the built methodology instance running the old class, with
    # `spec` naming one configuration while `_execute` runs another. The cache leans on
    # the same guarantee from the other side. `_ensure` short-circuits on a stored
    # fingerprint *because* a settled step cannot have changed.

    #: Attributes only `__init__` may write. A kind names its own and unions this, so a
    #: `Step` subclass keeps whatever attributes it likes. The guard protects what the
    #: plan machinery reads, not every attribute on the object. `parents` is not
    #: here: it is a property, which refuses assignment without any help.
    _READ_ONLY: ClassVar[frozenset[str]] = frozenset()

    def __setattr__(self, name: str, value: object) -> None:
        """Refuse writes to what a spec, or the plan's shape, is built from."""
        if name in self._READ_ONLY:
            raise AttributeError(
                f"{type(self).__name__}.{name} is read-only. A step is settled once "
                "built, so rebuild the plan to change it."
            )
        object.__setattr__(self, name, value)

    def _set(self, **attributes: object) -> None:
        """Write attributes past the read-only guard. `__init__` is the only caller."""
        for name, value in attributes.items():
            object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        """Return a short representation showing kind and collection state."""
        return f"<{type(self).__name__} {'collected' if self.is_collected else 'lazy'}>"

    def __str__(self) -> str:
        """How this step appears in a drawing."""
        return self.kind

    @property
    def is_collected(self) -> bool:
        """Whether this step has been materialised."""
        return self._fp is not None

    # -- plan identity ----------------------------------------------------------------

    @property
    @abstractmethod
    def spec(self) -> BaseModel:
        """This step's settings, as a serialisable model.

        One model per step kind, in `matchlab.specs`. It must carry everything this
        step's output depends on and nothing else. That is the invariant `_spec_key`
        rests on, and the one to check when adding a setting. Omit something that
        changes the output and collect will hand back a stale artifact without
        re-running (see `_fingerprint`).

        Specs describe a step's own settings, not its inputs'. Edges live on
        `parents`, and `_fingerprint` already folds in their fingerprints.
        """
        ...

    def _spec_key(self) -> bytes:
        """Bytes identifying this step's spec.

        `Source` extends this with a content hash of the data it read, so that the
        fingerprint tracks the warehouse. For every other kind the spec is the
        whole story.
        """
        return json.dumps(self.spec.model_dump(mode="json"), sort_keys=True).encode()

    def _fingerprint(self) -> Fingerprint:
        """Address this step by kind, spec and inputs.

        The key is derived from the *plan*, not from the step's output. That is what
        makes it computable before the step runs, which is what lets `_ensure` skip
        work. An output digest would only be knowable once the work was already done.
        Sources are the exception. They fold a hash of the data they read into
        `_spec_key`, because no spec reveals that the warehouse moved.

        The trade-off is that the key can disagree with the bytes in both directions.

        * the spec omits something that changes the output, a **stale hit**, because
          `_ensure` never runs the step and reads the old artifact. `SplinkLinker`
          without an explicit seed is a live example. The only defence is that every
          `_spec_key` covers everything its step's output depends on. Treat that as
          the invariant to protect when adding a step or a setting.
        * the spec includes something that doesn't change the output, a **spurious
          miss**, re-running the step and everything below it. No spec does this
          today, and the way to keep it that way is to record settings only. A spec
          that described an *input* would, since identity already arrives via the
          parent fingerprint. A setting that must point at an input points at its
          position, which is not redundant. It decides which input the setting
          applies to.

        There is also no early cutoff. A `Transform` whose SQL is reformatted but
        semantically unchanged invalidates the whole subtree beneath it.

        TODO(fingerprints): split this into an action key (plan-derived, as now)
        mapping to an output digest (content-derived, recorded by the adapter on
        write), and build a child's key from its parents' output digests rather than
        their action keys. That buys early cutoff, storage dedup and rename tolerance.
        It does **not** fix stale hits. An output digest governs propagation, never
        admission, so a step whose action key is a hit still never runs and its digest
        is never consulted. Costs an indirection table in the adapter contract, a hash
        of every artifact on write, and the loss of a work-set knowable before running.
        See PLAN.md, "Known limitations", for the full ledger.
        """
        parts: list[bytes] = [self.kind.encode(), self._spec_key()]
        for parent in self.parents:
            if parent._fp is None:  # collect orders upstream first
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

        A step this object already collected short-circuits on its stored `_fp`, without
        recomputing the fingerprint. A settled step cannot have changed, so recomputing
        would only confirm the fingerprint already held. That saves a spec hash per step
        per collect, which for a `Source` means re-hashing every row hash it memoised.

        This classifies the outcome but reports none of it. `collect` holds the walk,
        and therefore each step's position, the thing a record has to quote to be
        findable in the plan. Reporting belongs there, with the single
        `matchlab.progress.Progress` that owns what is drawn and what is logged. That is
        also what keeps the two from ever disagreeing. This keeps to the work.

        Returns:
            What it took. `DONE` if this call computed the step, `CACHED` if the
            artifact was already stored.
        """
        self._adapter = adapter

        if self._fp is not None and adapter.has(self._fp):
            return StepStatus.CACHED

        fp = self._fingerprint()

        if adapter.has(fp):  # cache hit, skip the work entirely
            self._fp = fp
            return StepStatus.CACHED

        self._execute(adapter, fp)
        self._fp = fp
        return StepStatus.DONE

    # -- public API -------------------------------------------------------------------

    def collect(
        self, adapter: Adapter | None = None, interactive: bool | None = None
    ) -> Self:
        """Materialise this step and everything it depends on.

        Steps whose artifact is already stored are skipped without being run, so
        re-collecting after adding a downstream step only does the new work.

        Args:
            adapter: Where to read and write artifacts. Defaults to the module-level
                adapter (a DuckDB store in the user cache directory).
            interactive: Whether someone is watching. `None`, the default, takes a
                terminal or a notebook as a yes. When they are, the plan is drawn as a
                live tree redrawn in place, and not logged. The tree on screen is the
                key those `[step N]` records need, and it stays there. When they are
                not, the plan is logged instead. See `matchlab.progress`.

        Returns:
            This step, now collected.
        """
        adapter = adapter or default_adapter()
        # One walk, used for both. It fixes each step's position, so a `step 7` the
        # reporter logs is the node it drew as `[7]`.
        steps = lineage.walk(self)
        with report(self, steps, interactive, adapter) as reporter:
            for step in steps:
                reporter.begin(step)
                try:
                    status = step._ensure(adapter)
                except Exception:
                    reporter.end(step, StepStatus.FAILED)
                    raise
                reporter.end(step, status)
        return self

    def lineage(self) -> list["Step"]:
        """Return this step and all its inputs, upstream-first."""
        return lineage.walk(self)

    def draw(self) -> str:
        """Render this step's sub-plan as a tree."""
        return lineage.draw(self)

    def fingerprints(self) -> set[Fingerprint]:
        """Address every artifact this plan is made of, its own and its inputs'.

        Which artifacts belong to a plan is the plan's own business, so this is where
        a store gets told: `adapter.prune(keep=plan.fingerprints())` hands storage a set
        of addresses it already understands, rather than a graph it would have to learn
        to walk.

        Returns:
            One fingerprint per step in `lineage()`. A set, because two steps in one
            plan can address the same artifact. Identical specs over identical
            inputs is the same bytes, and it is stored once.

        Raises:
            RuntimeError: If any step has not been collected. An uncollected plan names
                no artifacts at all, so answering with a smaller set would quietly tell
                a caller that less is worth keeping than they think.
        """
        return {step._collected()[1] for step in self.lineage()}

    # -- helpers for subclasses -------------------------------------------------------

    def _collected(self) -> tuple[Adapter, Fingerprint]:
        """Return the adapter this step was collected into, and its fingerprint.

        Both together, because everything that reads a stored artifact needs both and
        neither exists before collection. Returning the pair is what lets a caller
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
