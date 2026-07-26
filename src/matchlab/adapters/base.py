"""The storage adapter contract for matchlab.

An adapter is **storage, not an engine**. It persists the artifacts each collected DAG
step produces, keyed by that step's content fingerprint, and reads them back. It does
*not* resolve anything on demand — the server's `_build_unified_query` is gone.
Resolvers materialise their complete, merge-forward resolution at collect time (see
`spikes/phase0_materialize_forward.py`) and hand the adapter a finished table.

Artifacts, by step kind (schemas from `matchlab.core.arrow`):

* Source   → warehouse extract (arbitrary schema) + leaf assignment `(key, leaf)`.
* View     → the materialised view (arbitrary schema), only when collected directly.
* Model    → edge list, `SCHEMA_MODEL_EDGES` `(left_id, right_id, score)`.
* Resolver → complete flat resolution, `SCHEMA_EVAL_SAMPLES` `(root, leaf, key, src)`.
             This is `merge(upstream complete resolution, own clusters)` — the Phase 0
             finding — NOT just the resolver's own clusters.

Plus evaluation storage (judgements + cluster expansion), publication (`publish` points
a label at a resolution) and `close`.

**Nothing here deletes an artifact on the store's own initiative.** A store keeps what
it is given until the owner disposes of it — see the guide's "Reclaiming storage".
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

import polars as pl

from matchlab.core.eval import Judgement

Fingerprint = bytes


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
        """Store a materialised view (arbitrary schema).

        Views are fused by default; this is only called when one is collected
        directly, so downstream steps read it instead of recomputing.
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
            resolution: `SCHEMA_EVAL_SAMPLES` columns `(root, leaf, key, source)`,
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

        Returns `SCHEMA_EVAL_SAMPLES` rows `(root, leaf, key, source)` for the sampled
        roots.
        """
        ...

    # -- lifecycle --------------------------------------------------------------------

    def close(self) -> None:  # noqa: B027 - optional concrete hook, not abstract
        """Release any underlying resources. Override if needed."""
