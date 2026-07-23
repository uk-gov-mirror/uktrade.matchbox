"""The storage adapter contract for local-only Matchbox / matchlab.

An adapter is **storage, not an engine**. It persists the artifacts each collected DAG
step produces, keyed by that step's content fingerprint, and reads them back. It does
*not* resolve anything on demand — the server's `_build_unified_query` is gone.
Resolvers materialise their complete, merge-forward resolution at collect time (see
`spikes/phase0_materialize_forward.py`) and hand the adapter a finished table.

Artifacts, by step kind (schemas from `matchbox.common.arrow`):

* Source   → warehouse extract (arbitrary schema) + leaf assignment `(key, leaf)`.
* Model    → edge list, `SCHEMA_MODEL_EDGES` `(left_id, right_id, score)`.
* Resolver → complete flat resolution, `SCHEMA_EVAL_SAMPLES` `(root, leaf, key, src)`.
             This is `merge(upstream complete resolution, own clusters)` — the Phase 0
             finding — NOT just the resolver's own clusters.

Plus evaluation storage (judgements + cluster expansion) and lifecycle (`gc`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl

from matchbox.common.eval import Judgement

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
        extract: pl.DataFrame,
        leaves: pl.DataFrame,
    ) -> None:
        """Store a collected source.

        Args:
            fp: The source step's fingerprint.
            name: The source's DAG name (used to tag its rows in resolutions).
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

    # -- resolvers --------------------------------------------------------------------

    @abstractmethod
    def store_resolver(self, fp: Fingerprint, resolution: pl.DataFrame) -> None:
        """Store a resolver's complete flat resolution.

        Args:
            fp: The resolver step's fingerprint.
            resolution: `SCHEMA_EVAL_SAMPLES` columns `(root, leaf, key, source)`,
                already merged forward over all upstream leaves. The adapter does not
                verify the merge — that is the client's contract — but it does validate
                the schema.
        """
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
        """Return `(judgements, expansion)` tables for `matchbox.common.eval`.

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
