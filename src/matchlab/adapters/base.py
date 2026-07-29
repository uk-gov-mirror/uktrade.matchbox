"""The storage adapter contract for matchlab.

An adapter is **storage, not an engine**. It persists the artifacts each collected DAG
step produces, keyed by that step's content fingerprint, and reads them back. It does
*not* resolve anything on demand — the server's `_build_unified_query` is gone.
Resolvers materialise their complete, merge-forward resolution at collect time and hand
the adapter a finished table.

Artifacts, by step kind (schemas in `matchlab.core.schemas`, which holds exactly the
shapes that cross this boundary):

* Source   → warehouse extract (arbitrary schema) + leaf assignment `(key, leaf)`.
* View     → the cleaned view (arbitrary schema).
* Model    → edge list, `SCHEMA_MODEL_EDGES` `(left_id, right_id, score)`.
* Resolver → complete flat resolution, `SCHEMA_RESOLUTION` `(root, leaf, key, src)`.
             This is `merge(upstream complete resolution, own clusters)` — the Phase 0
             finding — NOT just the resolver's own clusters.

Plus evaluation storage (judgements + cluster expansion), publication (`publish` points
a label at a resolution) and `close`.

**Nothing here deletes an artifact on the store's own initiative.** A store keeps what
it is given until the owner disposes of it, which is what `trim` is: it deletes only
what the caller has said it may, and never a published resolution. See the guide's
"Reclaiming storage".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from matchlab.core.kinds import StepKind
from matchlab.eval.judgements import Judgement

Fingerprint = bytes


class StoreStats(BaseModel):
    """What a store holds, and what it costs.

    A store keeps everything collected into it, so it grows with every plan and every
    edit to one. Adapters subclass this to report metrics that are relevant to them.

    Attributes:
        location: Where the store is, as a reader would name it — a path, a URI,
            `":memory:"`. Always present, because a store you cannot point at is one
            you cannot go and delete.
        bytes: The store's size. For anything file-backed that is a **high-water
            mark**: a store that briefly held 4 GB reports 4 GB after the rows are
            gone, because deleting inside a database file does not return space to the
            OS. That is the right number for a disk filling up, and the wrong one for
            "how much data do I have".
        artifacts: How many artifacts of each step kind are stored.
        labels: How many published labels point into the store.
    """

    model_config = ConfigDict(frozen=True)

    location: str
    bytes: int
    artifacts: dict[StepKind, int] = Field(default_factory=dict)
    labels: int = 0

    @property
    def size(self) -> str:
        """This store's size, as a phrase.

        A hook, because a bare figure can mislead: bytes held in memory and bytes
        written to a disk are not the same claim, and only the backend knows which it
        just reported.
        """
        return format_bytes(self.bytes)

    def describe(self, since: StoreStats | None = None) -> str:
        """One clause saying what the store costs, for a collect to print.

        Args:
            since: An earlier reading, so the growth between them can be attributed to
                whatever happened in between. Without it the figure is the store's size
                and nothing more, which names no cause.
        """
        count = sum(self.artifacts.values())
        growth = (
            f" ({format_bytes(self.bytes - since.bytes, signed=True)})"
            if since is not None
            else ""
        )
        return f"Store {self.size}{growth}, {count} artifact{'' if count == 1 else 's'}"


class TrimResult(BaseModel):
    """What a trim removed, and what that actually recovered.

    Adapters subclass this where they have more to say.

    Attributes:
        removed: How many artifacts were deleted.
        kept: How many survived — the ones named, the ones their lineage needed, and
            everything published.
        reclaimed: Bytes genuinely returned. **Measured** as the store's size before
            minus after, not totted up from what was deleted. The two are not the same
            number: deleting inside a database file usually frees nothing at all until
            the file is rewritten, and a reclaim that reported the bytes it *deleted*
            would claim to have freed space while the disk sat unchanged.
    """

    model_config = ConfigDict(frozen=True)

    removed: int
    kept: int
    reclaimed: int

    def describe(self) -> str:
        """One line saying what happened, for a caller to print."""
        return (
            f"Removed {self.removed} artifact{'' if self.removed == 1 else 's'}, "
            f"kept {self.kept}, reclaimed {format_bytes(self.reclaimed)}"
        )


def format_bytes(count: int, *, signed: bool = False) -> str:
    """Render a byte count the way someone reading a disk would.

    Binary units, because that is what a file browser and `du -h` report, and a size
    that disagrees with the one the user can check is worse than no size at all.
    """
    sign = "+" if signed and count >= 0 else "-" if count < 0 else ""
    size = float(abs(count))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            # Whole bytes never want a decimal point; larger units always do, so a
            # store that grew stays visibly distinct from one that didn't.
            precision = 0 if unit == "B" else 1
            return f"{sign}{size:.{precision}f} {unit}"
        size /= 1024
    raise AssertionError("unreachable: the loop returns at TB")  # pragma: no cover


class Adapter(ABC):
    """Fingerprint-keyed storage for collected DAG-step artifacts.

    Implementations are single-user and local. `DuckDBAdapter` is the reference.
    """

    # -- existence / caching ----------------------------------------------------------

    @abstractmethod
    def has(self, fp: Fingerprint) -> bool:
        """Return whether an artifact for this fingerprint is already stored.

        The client uses this to skip re-running a step whose plan is unchanged.
        """
        ...

    # -- sources ----------------------------------------------------------------------

    @abstractmethod
    def store_source(
        self,
        fp: Fingerprint,
        key_field: str,
        extract: pl.DataFrame,
        leaves: pl.DataFrame,
    ) -> None:
        """Store a collected source.

        Args:
            fp: The source step's fingerprint.
            key_field: Which column of `extract` holds the key. Stored so the extract
                can be read back and joined to a resolution without the plan.
            extract: The warehouse extract (arbitrary schema) to cache.
            leaves: The leaf assignment, columns `(key: str, leaf: uint64)`.
        """
        ...

    @abstractmethod
    def read_source_extract(self, fp: Fingerprint) -> pl.DataFrame:
        """Return the cached warehouse extract for a stored source."""
        ...

    @abstractmethod
    def read_source_leaves(self, fp: Fingerprint) -> pl.DataFrame:
        """Return the `(key, leaf)` assignment for a stored source."""
        ...

    # -- identifiers ------------------------------------------------------------------

    @abstractmethod
    def read_identifiers(
        self,
        source_fp: Fingerprint,
        source_name: str,
        resolver_fp: Fingerprint | None = None,
    ) -> pl.DataFrame:
        """Return `(id, source, key, leaf)` for one source's records.

        `id` is `resolver_fp`'s root cluster when reading through a resolver, otherwise
        the source's own leaf. This is the *upstream resolution* a downstream resolver
        needs in order to carry every reachable leaf forward, including records no model
        matched.

        A query rather than an artifact. Nothing here is computed: both
        readings are projections of tables this store already holds — `source_leaves`
        and `resolution`. There is correspondingly nothing to cache, and caching it
        under a view's fingerprint would be wrong anyway, since what is returned depends
        only on the source and resolver read and not on how a view cleans them.

        Args:
            source_fp: Fingerprint of the stored source whose records are wanted.
            source_name: The source's name, which is returned in the `source` column
                and tags each row for the resolution below.
            resolver_fp: Fingerprint of the resolver to read through, or `None` to read
                the source's own leaves.
        """
        ...

    # -- models -----------------------------------------------------------------------

    @abstractmethod
    def store_model(self, fp: Fingerprint, edges: pl.DataFrame) -> None:
        """Store a model's edge list (`SCHEMA_MODEL_EDGES`)."""
        ...

    @abstractmethod
    def read_model(self, fp: Fingerprint) -> pl.DataFrame:
        """Return a stored model's edge list."""
        ...

    # -- views ------------------------------------------------------------------------

    @abstractmethod
    def store_view(self, fp: Fingerprint, table: pl.DataFrame) -> None:
        """Store a cleaned view (arbitrary schema).

        Called on every collect, so a view feeding several models is computed once and
        read back by each of them.
        """
        ...

    @abstractmethod
    def read_view(self, fp: Fingerprint) -> pl.DataFrame:
        """Return a stored view."""
        ...

    # -- resolvers --------------------------------------------------------------------

    @abstractmethod
    def store_resolver(
        self,
        fp: Fingerprint,
        resolution: pl.DataFrame,
        sources: Mapping[str, Fingerprint] | None = None,
    ) -> None:
        """Store a resolver's complete flat resolution.

        Args:
            fp: The resolver step's fingerprint.
            resolution: `SCHEMA_RESOLUTION` columns `(root, leaf, key, source)`,
                already merged forward over all upstream leaves. The adapter does not
                verify the merge — that is the client's contract — but it does validate
                the schema.
            sources: Source name to fingerprint, for every source this resolution
                covers. A resolution names its sources but one store can hold several
                generations of a name, so this records which were actually used.
        """
        ...

    # -- labels -----------------------------------------------------------------------
    #
    # A **label** is a pointer, kept here, from a string you chose to a resolution you
    # want to find again. It is deliberately not called a name: a *name* belongs to a
    # source and is part of its output, while a label belongs to the store and is part
    # of finding things in it. Storing an artifact and labelling one are separate acts —
    # `Resolver.publish` does the second, after the first.
    #
    # This is what lets evaluation run against a store alone, with no plan.

    @abstractmethod
    def publish(self, label: str, fp: Fingerprint) -> None:
        """Point `label` at `fp`, replacing whatever it pointed at before.

        The adapter moves the pointer without arguing; whether overwriting is allowed
        is decided by the caller, which knows what the user asked for.
        """
        ...

    @abstractmethod
    def find(self, label: str) -> Fingerprint | None:
        """Return the fingerprint a label points at, if any."""
        ...

    @abstractmethod
    def labels(self) -> list[str]:
        """Return every label in this store, sorted."""
        ...

    @abstractmethod
    def source_key_field(self, fp: Fingerprint) -> str:
        """Return which column of a stored source's extract holds the key."""
        ...

    @abstractmethod
    def resolution_sources(self, fp: Fingerprint) -> dict[str, Fingerprint]:
        """Return source name to fingerprint for a stored resolution."""
        ...

    @abstractmethod
    def read_resolver(self, fp: Fingerprint) -> pl.DataFrame:
        """Return a resolver's stored resolution `(root, leaf, key, source)`."""
        ...

    # -- evaluation -------------------------------------------------------------------

    @abstractmethod
    def store_judgement(self, judgement: Judgement, user_name: str = "local") -> None:
        """Persist a user judgement, expanding its clusters to leaves for scoring."""
        ...

    @abstractmethod
    def read_eval_data(
        self, tag: str | None = None
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Return `(judgements, expansion)` tables for `matchlab.core.eval`.

        `judgements` follows `SCHEMA_JUDGEMENTS`; `expansion` follows
        `SCHEMA_CLUSTER_EXPANSION`. Filtered to `tag` when given.
        """
        ...

    @abstractmethod
    def sample(
        self, resolver_fp: Fingerprint, n: int, seed: int | None = None
    ) -> pl.DataFrame:
        """Sample up to `n` clusters from a stored resolution for evaluation.

        Returns `SCHEMA_RESOLUTION` rows `(root, leaf, key, source)` for the sampled
        roots.
        """
        ...

    # -- introspection ----------------------------------------------------------------

    @abstractmethod
    def stats(self) -> StoreStats:
        """Report the store's size and contents.

        Abstract rather than an optional hook like `close`, because there is no honest
        default. A store that reported zero would be wrong in exactly the place a user
        is being asked to trust a number, and a store that cannot say how much room it
        is taking is not one to hand a growing cache to.

        Called on every collect, so it must be cheap. It may also settle pending writes
        in order to report a size that has stopped moving — `DuckDBAdapter` checkpoints
        — so this is not guaranteed to be a pure read.
        """
        ...

    # -- maintenance ------------------------------------------------------------------

    @abstractmethod
    def trim(self, keep: Iterable[Fingerprint | str] = ()) -> TrimResult:
        """Delete every artifact except the ones named, and reclaim what that frees.

        **You say what to keep; nothing is inferred.** This is deliberately not the
        `gc()` that used to live here. That one worked out what was still alive from
        which Python objects happened to be reachable, so a fresh interpreter — where
        nothing is reachable — considered the whole store garbage and duly emptied one
        that had a published resolution in it. An artifact's worth has nothing to do
        with whether some process is holding the variable that produced it. So the root
        set arrives as an argument, from a caller who has just named it.

        Both kinds of name are ones a store already understands. A plan is not: which
        artifacts belong to one is the plan's business, and `Step.fingerprints()` is
        where it answers. Storage should not have to learn to walk a graph in order to
        know what it may delete.

        **Published labels are kept whether or not they are named**, along with the
        sources their resolutions need in order to stay readable. Publishing is the
        strongest "keep this" the system has, and losing one because a caller forgot to
        list it would be the old mistake in new clothes.

        Implementations must:

        * never touch stored judgements, which are human work and cannot be recomputed
          from anything;
        * report `reclaimed` as space genuinely returned, measured rather than assumed.
          Deleting is not always the same as reclaiming, and a store that says it freed
          something it did not is worse than one that frees nothing.

        Args:
            keep: What to preserve — a `Fingerprint`, or a `str` naming a published
                label, which keeps that resolution and the sources it reads through.

        Returns:
            What was removed and what that recovered.

        Raises:
            ValueError: If this would empty the store — an empty `keep` with nothing
                published. Deleting everything is deleting the file, and should not be
                what an accidentally-empty list does.
        """
        ...

    # -- lifecycle --------------------------------------------------------------------

    def close(self) -> None:  # noqa: B027 - optional concrete hook, not abstract
        """Release any underlying resources. Override if needed."""
