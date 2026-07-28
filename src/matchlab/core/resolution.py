"""Client-side resolution — the local replacement for the server's query engine.

The server (`server/postgresql/utils/query.py:_build_unified_query`) resolved source
keys to root cluster IDs *on demand*, projecting up the resolver hierarchy with a
priority-`COALESCE` at query time. Locally we do the opposite: each resolver, when it
runs, materialises its **complete, merge-forward** resolution once, and everything
downstream (`get_matches`, `lookup_key`, models querying through it) just reads it.

This module holds the pure functions that produce that resolution:

* `leaf_id` / `root_id` — client-side, content-addressed cluster IDs. A leaf is a hash
  of a record's content, which is the correct anchor for evaluation, not merely a
  reproducibility trick — see `leaf_id` for the full reasoning.
* `materialise_resolution` — turn the connected-component clusters a resolver computed
  (over query-space IDs, covering only IDs its models formed edges over) plus the
  upstream resolution (every reachable ID → its `(source, key, leaf)` rows) into the
  complete `(root, leaf, key, source)` table.

The merge-forward requirement — that leaves grouped upstream but untouched by this
resolver inherit their upstream cluster rather than collapsing to singletons — is the
Phase 0 finding. It is implemented here by
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
import polars_hash  # noqa: F401 - registers the `nchash` expression namespace

from matchlab.core.schemas import SCHEMA_RESOLUTION

# Upstream resolution schema consumed by `materialise_resolution`: every reachable
# query-space ID mapped to a source row and its leaf.
UPSTREAM_COLUMNS = ("id", "source", "key", "leaf")


def leaf_id(row_hash: pl.Expr) -> pl.Expr:
    """Map a column of row content hashes to stable 64-bit leaf cluster IDs.

    A record's identity is a hash of its **content**, not of its key. This is the
    load-bearing decision, and it is deliberate — not a leftover from the server, which
    hashed to save storage. The reason is evaluation:

    A judgement is a person or a model saying "looking at *this*, these records are the
    same entity." The only input to that decision is the content shown — a name, a
    postcode. The key plays no part; the judge never sees it. So a judgement must be
    anchored to the content it was actually made against:

    * Anchor to the **key** and a judgement outlives a content change. "acme / london =
      acme / london", decided against evidence that has since become "acme /
      manchester", now silently stands — a decision attributed to evidence the judge
      never saw. It looks inexplicable later, or quietly validates a match nobody made.
    * Anchor to the **content** (this) and the leaf ID changes when the content does,
      so the old judgement stops applying — correctly. Nobody has judged the new
      content; there should be no judgement about it. Judgements decay exactly when
      their evidence does.

    The same logic explains why identical rows share a leaf: to any judge they are
    indistinguishable, so there is no decision to make. Differing keys don't separate
    them, because a key is not evidence.

    **The invariant this rests on: what is hashed must equal what is shown.** Hash a
    column the sampler doesn't display and a judgement decays for a reason the judge
    can't see; display one that isn't hashed and a judgement survives a change the judge
    can see. They line up today — `leaf_id` covers every non-key column (see
    `Source._read_warehouse`) and `get_samples` shows every non-key column — and "the
    extract is the whole declaration" is what keeps them lined up: select a column and
    it is both hashed and shown.

    Content-addressing also makes runs reproducible and IDs stable across re-collect,
    but that is a consequence, not the reason.

    Vectorised deliberately: this runs once per source row, and a Python function called
    through `map_elements` dominated collection.
    """
    return row_hash.bin.slice(0, 8).bin.reinterpret(dtype=pl.UInt64, endianness="big")


def root_id(leaves: pl.Expr) -> pl.Expr:
    """Deterministic 64-bit root cluster ID for a column of leaf lists.

    Invariant to leaf order — callers sort — and to the (arbitrary) component label,
    so two runs that produce the same clustering produce the same root IDs.

    Vectorised for the same reason as `leaf_id`: one call per cluster is one Python
    call per cluster, and there are as many clusters as there are entities.
    """
    # `polars_hash` registers the `nchash` namespace on polars expressions at import.
    joined = leaves.cast(pl.List(pl.Utf8)).list.join(",")
    return joined.nchash.xxh3_64().cast(pl.UInt64)


def root_id_of(leaves: Iterable[int]) -> int:
    """`root_id` for a single leaf set. For tests and one-off checks, not for loops."""
    leaves = sorted(leaves)
    if not leaves:
        raise ValueError("A cluster must contain at least one leaf")
    frame = pl.DataFrame({"leaves": [leaves]}).with_columns(
        pl.col("leaves").list.eval(pl.element().cast(pl.UInt64))
    )
    return frame.select(root_id(pl.col("leaves")).alias("root"))["root"][0]


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
        `SCHEMA_RESOLUTION` `(root, leaf, key, source)`: one row per reachable source
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
        return pl.from_arrow(SCHEMA_RESOLUTION.empty_table())

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
        .with_columns(root_id(pl.col("_leaves")).alias("root"))
        .select("_component", "root")
    )

    return (
        labelled.join(component_roots, on="_component")
        .select("root", "leaf", "key", "source")
        .unique()
        .sort("root", "leaf", "key")
    )
