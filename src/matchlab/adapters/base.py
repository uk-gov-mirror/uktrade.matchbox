"""The storage adapter contract for matchlab.

An adapter is **storage, not an engine**. It persists the artifacts each collected DAG
step produces, keyed by that step's content fingerprint, and reads them back. It does
*not* resolve anything on demand — the server's `_build_unified_query` is gone.
Resolvers materialise their complete, merge-forward resolution at collect time (see
`spikes/phase0_materialize_forward.py`) and hand the adapter a finished table.

Artifacts, by step kind (schemas from `matchlab.core.arrow`):

* Source   → warehouse extract (arbitrary schema) + leaf assignment `(key, leaf)`.
* Model    → edge list, `SCHEMA_MODEL_EDGES` `(left_id, right_id, score)`.
* Resolver → complete flat resolution, `SCHEMA_EVAL_SAMPLES` `(root, leaf, key, src)`.
             This is `merge(upstream complete resolution, own clusters)` — the Phase 0
             finding — NOT just the resolver's own clusters.

Plus evaluation storage (judgements + cluster expansion) and lifecycle (`gc`).
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
        name: str,
        key_field: str,
        extract: pl.DataFrame,
        leaves: pl.DataFrame,
    ) -> None:
        """Store a collected source.

        Args:
            fp: The source step's fingerprint.
            name: The source's name (used to tag its rows in resolutions).
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
    def store_model(self, fp: Fingerprint, name: str, edges: pl.DataFrame) -> None:
        """Store a model's edge list (`SCHEMA_MODEL_EDGES`)."""
        ...

    @abstractmethod
    def read_model(self, fp: Fingerprint) -> pl.DataFrame:
        """Return a stored model's edge list."""
        ...

    # -- cleaned views ----------------------------------------------------------------

    @abstractmethod
    def store_clean(self, fp: Fingerprint, table: pl.DataFrame) -> None:
        """Store a materialised cleaned view (arbitrary schema).

        Cleaned views are fused by default; this is only called when one is collected
        directly, so downstream steps read it instead of recomputing.
        """
        ...

    @abstractmethod
    def read_clean(self, fp: Fingerprint) -> pl.DataFrame:
        """Return a stored cleaned view."""
        ...

    # -- resolvers --------------------------------------------------------------------

    @abstractmethod
    def store_resolver(
        self,
        fp: Fingerprint,
        name: str,
        resolution: pl.DataFrame,
        sources: Mapping[str, Fingerprint] | None = None,
    ) -> None:
        """Store a resolver's complete flat resolution.

        Args:
            fp: The resolver step's fingerprint.
            name: The resolver's name, so a store can be browsed without the plan.
            resolution: `SCHEMA_EVAL_SAMPLES` columns `(root, leaf, key, source)`,
                already merged forward over all upstream leaves. The adapter does not
                verify the merge — that is the client's contract — but it does validate
                the schema.
            sources: Source name to fingerprint, for every source this resolution
                covers. A resolution names its sources but one store can hold several
                generations of a name, so this records which were actually used.
        """
        ...

    # -- lookups ----------------------------------------------------------------------
    #
    # Enough to read a stored resolution back without the plan that built it, which is
    # what lets evaluation run against a store alone.

    @abstractmethod
    def find(self, kind: str, name: str) -> Fingerprint | None:
        """Return the fingerprint of a stored artifact by kind and name."""
        ...

    @abstractmethod
    def names(self, kind: str) -> list[str]:
        """Return the names of stored artifacts of a kind, sorted."""
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

    @abstractmethod
    def gc(self, live: set[Fingerprint]) -> int:
        """Drop every stored artifact whose fingerprint is not in `live`.

        Returns the number of artifacts removed.
        """
        ...

    def close(self) -> None:  # noqa: B027 - optional concrete hook, not abstract
        """Release any underlying resources. Override if needed."""
