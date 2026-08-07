"""Each resolver materialises its complete, merge-forward output once.

Every resolver, when it runs, computes that output and stores it. Everything
downstream (`entities`, `lookup_key`, models querying through it) just reads the
stored table.

This module holds the pure functions that produce that output:

* `leaf_id` / `root_id` — content-addressed cluster IDs. A leaf is a hash of a
  record's content. That is the correct anchor for evaluation, not merely a
  reproducibility trick — see `leaf_id` for the full reasoning.
* `materialise_resolver_output` — turns a resolver's clusters and upstream output
  into the complete result table. See its own docstring for the shape of each.

**Merge-forward** means a leaf grouped upstream but untouched by this resolver
inherits its upstream cluster rather than collapsing to a singleton.
`materialise_resolver_output` implements this by giving every *untouched* reachable ID
its own component ("fall-through"), so the upstream grouping survives.

This module assumes a resolver's inputs share a consistent lineage, so each
reachable leaf maps to exactly one upstream ID. DAG construction enforces that, so
nothing here needs to reconcile a leaf that maps to more than one.
"""

from collections.abc import Iterable

import polars as pl
import polars_hash  # noqa: F401 - registers the `nchash` expression namespace

from matchlab.core.schemas import SCHEMA_RESOLVER_OUTPUT

# Every reachable query-space ID, mapped to a source row and its leaf. This is the
# shape `materialise_resolver_output` expects for its `upstream` argument.
UPSTREAM_COLUMNS = ("id", "source", "key", "leaf")


def leaf_id(row_hash: pl.Expr) -> pl.Expr:
    """Map a column of row content hashes to stable 64-bit leaf cluster IDs.

    A record's identity is a hash of its **content**, not of its key. This is the
    load-bearing decision, and it is deliberate, not an accident of implementation.
    The reason is evaluation.

    A judgement is a person or a model saying "looking at *this*, these records are the
    same entity." The only input to that decision is the content shown — a name, a
    postcode. The key plays no part. The judge never sees it. A judgement must
    therefore be anchored to the content it was actually made against:

    * If you anchor to the **key**, a judgement outlives a content change. Say a
      judgement was made when the evidence read "acme / london = acme / london". If
      that evidence later becomes "acme / manchester", the judgement still stands —
      attributed to evidence the judge never saw. It looks inexplicable later, or
      quietly validates a match nobody made.
    * If you anchor to the **content** instead (this module's choice), the leaf ID
      changes whenever the content does, so the old judgement stops applying. That is
      correct. Nobody has judged the new content, so there should be no judgement
      about it. Judgements decay exactly when their evidence does.

    The same logic explains why identical rows share a leaf. To any judge they are
    indistinguishable, so there is no decision to make. Differing keys don't separate
    them, because a key is not evidence.

    **What gets hashed must match what gets shown.** If a column feeds the hash but
    the sampler doesn't display it, a judgement decays for a reason the judge never
    saw. If a column appears in the sampler but doesn't feed the hash, a judgement
    survives a change the judge did see, which is wrong.

    The two stay in sync today because both read the same definition. `leaf_id`
    hashes every non-key column (see `Source._read_warehouse`), and `get_samples`
    displays every non-key column. Selecting a column in the extract is what makes it
    both hashed and shown — there is no second list to keep in step by hand (see
    `Source`'s "the extract is the whole declaration").

    Content-addressing also makes runs reproducible and IDs stable across re-collect,
    but that is a consequence, not the reason.

    This is vectorised deliberately. Calling a Python function once per row, as
    `map_elements` would, dominates collection time.
    """
    return row_hash.bin.slice(0, 8).bin.reinterpret(dtype=pl.UInt64, endianness="big")


def root_id(leaves: pl.Expr) -> pl.Expr:
    """Deterministic 64-bit root cluster ID for a column of leaf lists.

    This is invariant to leaf order (callers sort) and to the arbitrary component
    label, so two runs that produce the same clustering produce the same root IDs.

    This is vectorised for the same reason as `leaf_id`. Calling this once per
    cluster, in Python, would mean as many calls as there are entities.
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


def materialise_resolver_output(
    clusters: pl.DataFrame,
    upstream: pl.DataFrame,
) -> pl.DataFrame:
    """Build a resolver's complete, merge-forward output.

    Args:
        clusters: `SCHEMA_CLUSTERS` `(parent_id, child_id)` from the resolver's
            methodology. `child_id`s are query-space IDs the models formed edges over.
            IDs not present here are untouched by this resolver.
        upstream: The union of the resolver's input queries as `(id, source, key,
            leaf)` rows, covering *every* leaf reachable by the resolver, whether or
            not an edge formed over it. `id` is the query-space ID the model
            referenced.

    Returns:
        One row per reachable source record, shaped as `SCHEMA_RESOLVER_OUTPUT`
        `(root, leaf, key, source)`, with `root` the cluster it resolves to. Complete
        and merge-forward.
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
        return pl.from_arrow(SCHEMA_RESOLVER_OUTPUT.empty_table())

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
