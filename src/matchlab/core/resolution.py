"""Client-side resolution — the local replacement for the server's query engine.

The server (`server/postgresql/utils/query.py:_build_unified_query`) resolved source
keys to root cluster IDs *on demand*, projecting up the resolver hierarchy with a
priority-`COALESCE` at query time. Locally we do the opposite: each resolver, when it
runs, materialises its **complete, merge-forward** resolution once, and everything
downstream (`get_matches`, `lookup_key`, models querying through it) just reads it.

This module holds the pure functions that produce that resolution:

* `leaf_id` / `root_id` — client-side, content-addressed cluster IDs (the server used
  to mint these as sequences; locally they are deterministic hashes so runs are
  reproducible and cacheable).
* `materialise_resolution` — turn the connected-component clusters a resolver computed
  (over query-space IDs, covering only IDs its models formed edges over) plus the
  upstream resolution (every reachable ID → its `(source, key, leaf)` rows) into the
  complete `(root, leaf, key, source)` table.

The merge-forward requirement — that leaves grouped upstream but untouched by this
resolver inherit their upstream cluster rather than collapsing to singletons — is the
Phase 0 finding (`spikes/phase0_materialize_forward.py`). It is implemented here by
giving every *untouched* reachable ID its own component ("fall-through"), so the
upstream grouping survives.

Well-formedness assumption: within a single resolver, each reachable leaf maps to one
upstream ID (i.e. its inputs share a consistent lineage, which DAG construction
enforces). The server handled the pathological multi-mapping case via leaf-level
COALESCE; the client does not need to, and does not.
"""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from matchlab.core.arrow import SCHEMA_EVAL_SAMPLES
from matchlab.core.hash import hash_values

# Upstream resolution schema consumed by `materialise_resolution`: every reachable
# query-space ID mapped to a source row and its leaf.
UPSTREAM_COLUMNS = ("id", "source", "key", "leaf")


def leaf_id(row_hash: bytes) -> int:
    """Map a source row's content hash to a stable 64-bit leaf cluster ID.

    Replaces the server's per-row sequence assignment. Content-addressed, so identical
    rows share a leaf and re-runs are stable.
    """
    return int.from_bytes(row_hash[:8], "big")


def root_id(leaves: Iterable[int]) -> int:
    """Deterministic 64-bit root cluster ID for a set of leaves.

    Invariant to leaf order (`hash_values` sorts) and to the (arbitrary) component
    label, so two runs that produce the same clustering produce the same root IDs.
    """
    leaves = list(leaves)
    if not leaves:
        raise ValueError("A cluster must contain at least one leaf")
    return int.from_bytes(hash_values(*leaves)[:8], "big")


def materialise_resolution(
    clusters: pl.DataFrame,
    upstream: pl.DataFrame,
) -> pl.DataFrame:
    """Build a resolver's complete, merge-forward resolution.

    Args:
        clusters: `SCHEMA_CLUSTERS` `(parent_id, child_id)` from the resolver's
            methodology. `child_id`s are query-space IDs the models formed edges over;
            IDs not present here are untouched by this resolver.
        upstream: `(id, source, key, leaf)` — the union of the resolver's input queries,
            covering *every* leaf reachable by the resolver, whether or not an edge
            formed over it. `id` is the query-space ID the model referenced.

    Returns:
        `SCHEMA_EVAL_SAMPLES` `(root, leaf, key, source)`: one row per reachable source
        record, with `root` the cluster it resolves to. Complete and merge-forward.
    """
    missing = set(UPSTREAM_COLUMNS) - set(upstream.columns)
    if missing:
        raise ValueError(f"upstream is missing columns: {sorted(missing)}")

    upstream = upstream.select(
        pl.col("id").cast(pl.UInt64),
        pl.col("source").cast(pl.Utf8),
        pl.col("key").cast(pl.Utf8),
        pl.col("leaf").cast(pl.UInt64),
    )

    if upstream.height == 0:
        return pl.from_arrow(SCHEMA_EVAL_SAMPLES.empty_table())

    child_to_parent = clusters.select(
        pl.col("child_id").cast(pl.UInt64),
        pl.col("parent_id").cast(pl.UInt64),
    )

    # Component label per reachable row:
    #   * touched IDs (in a cluster)  -> "c{parent_id}"  (this resolver's new cluster)
    #   * untouched IDs               -> "s{id}"         (fall-through: keep upstream)
    # Prefixes keep the two integer spaces from colliding.
    labelled = upstream.join(
        child_to_parent, left_on="id", right_on="child_id", how="left"
    ).with_columns(
        pl.when(pl.col("parent_id").is_not_null())
        .then("c" + pl.col("parent_id").cast(pl.Utf8))
        .otherwise("s" + pl.col("id").cast(pl.Utf8))
        .alias("_component")
    )

    # Content-addressed root per component, derived from its full leaf set.
    component_roots = (
        labelled.group_by("_component")
        .agg(pl.col("leaf").unique().sort().alias("_leaves"))
        .with_columns(
            pl.col("_leaves")
            .map_elements(root_id, return_dtype=pl.UInt64)
            .alias("root")
        )
        .select("_component", "root")
    )

    return (
        labelled.join(component_roots, on="_component")
        .select("root", "leaf", "key", "source")
        .unique()
        .sort("root", "leaf", "key")
    )
