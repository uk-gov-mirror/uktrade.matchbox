"""Tests for client-side resolution (Phase 2 keystone).

These bridge the abstract Phase 0 spike (`spikes/phase0_materialize_forward.py`) to the
concrete `(clusters, upstream)` data shapes the client actually produces, and assert
the merge-forward resolution — including fall-through of upstream-grouped, locally
untouched leaves.
"""

import polars as pl

from matchlab.core.arrow import SCHEMA_CLUSTERS
from matchlab.core.resolution import (
    leaf_id,
    materialise_resolution,
    root_id_of,
)


def _clusters(pairs: list[tuple[int, int]]) -> pl.DataFrame:
    if not pairs:
        return pl.from_arrow(SCHEMA_CLUSTERS.empty_table())
    parents, children = zip(*pairs, strict=True)
    return pl.DataFrame(
        {"parent_id": parents, "child_id": children},
        schema={"parent_id": pl.UInt64, "child_id": pl.UInt64},
    )


def _upstream(rows: list[tuple[int, str, str, int]]) -> pl.DataFrame:
    ids, sources, keys, leaves = zip(*rows, strict=True)
    return pl.DataFrame(
        {"id": ids, "source": sources, "key": keys, "leaf": leaves},
        schema={"id": pl.UInt64, "source": pl.Utf8, "key": pl.Utf8, "leaf": pl.UInt64},
    )


def _partition(resolution: pl.DataFrame) -> set[frozenset[str]]:
    """Group keys by resolved root, forgetting the (arbitrary) root labels."""
    grouped = resolution.group_by("root").agg(pl.col("key"))
    return {frozenset(keys) for keys in grouped["key"].to_list()}


# -- content-addressed IDs ------------------------------------------------------------


def test_leaf_id_is_stable_and_int() -> None:
    h = b"\xab" * 32
    frame = pl.DataFrame({"hash": [h, h]})
    leaves = frame.select(leaf_id(pl.col("hash")).alias("leaf"))["leaf"]
    assert leaves[0] == leaves[1]
    assert leaves[0] == int.from_bytes(h[:8], "big")


def test_root_id_is_order_invariant() -> None:
    assert root_id_of([3, 1, 2]) == root_id_of([1, 2, 3])
    assert root_id_of([1, 2]) != root_id_of([1, 3])


# -- the layered fall-through scenario (concrete form of the Phase 0 spike) ------------


def test_layered_fallthrough_preserves_upstream_clusters() -> None:
    """Apex links A+B; C, grouped by an upstream dedupe, must survive untouched.

    Leaf IDs: a1=1 a2=2 a3=3 b1=4 b2=5 c1=6 c2=7.
    Upstream: dedupe_A grouped {a1,a2} -> id 101; dedupe_C grouped {c1,c2} -> id 103.
    Apex model edge: (101, 4) links the A-cluster with b1. Nothing else is touched.
    """
    upstream = _upstream(
        [
            (101, "A", "ka1", 1),
            (101, "A", "ka2", 2),
            (3, "A", "ka3", 3),
            (4, "B", "kb1", 4),
            (5, "B", "kb2", 5),
            (103, "C", "kc1", 6),
            (103, "C", "kc2", 7),
        ]
    )
    clusters = _clusters([(1, 101), (1, 4)])  # apex merges upstream-A with b1

    resolution = materialise_resolution(clusters, upstream)

    assert _partition(resolution) == {
        frozenset({"ka1", "ka2", "kb1"}),  # the new apex cluster
        frozenset({"ka3"}),  # untouched singleton
        frozenset({"kb2"}),  # untouched singleton
        frozenset({"kc1", "kc2"}),  # fall-through: dedupe_C survives
    }
    # Every reachable record appears exactly once.
    assert resolution.height == upstream.height
    assert set(resolution.columns) == {"root", "leaf", "key", "source"}
    assert resolution.schema["root"] == pl.UInt64


def test_grouped_ids_share_a_root_and_untouched_do_not() -> None:
    upstream = _upstream(
        [
            (101, "A", "ka1", 1),
            (101, "A", "ka2", 2),
            (103, "C", "kc1", 6),
            (103, "C", "kc2", 7),
        ]
    )
    clusters = _clusters([(1, 101), (1, 103)])  # link the two upstream clusters
    resolution = materialise_resolution(clusters, upstream)

    # All four keys resolve to a single root.
    assert resolution["root"].n_unique() == 1


# -- single resolver, no layering -----------------------------------------------------


def test_single_resolver_over_bare_leaves() -> None:
    # No upstream resolver: query-space id == leaf.
    upstream = _upstream(
        [
            (1, "A", "ka1", 1),
            (2, "A", "ka2", 2),
            (3, "A", "ka3", 3),
        ]
    )
    clusters = _clusters([(1, 1), (1, 2)])  # dedupe merges a1, a2
    resolution = materialise_resolution(clusters, upstream)

    assert _partition(resolution) == {
        frozenset({"ka1", "ka2"}),
        frozenset({"ka3"}),
    }


# -- determinism ----------------------------------------------------------------------


def test_roots_are_deterministic_across_runs() -> None:
    upstream = _upstream([(1, "A", "ka1", 1), (2, "A", "ka2", 2)])
    clusters = _clusters([(1, 1), (1, 2)])
    a = materialise_resolution(clusters, upstream)
    b = materialise_resolution(clusters, upstream)
    assert a.sort("leaf").equals(b.sort("leaf"))


# -- edge cases -----------------------------------------------------------------------


def test_empty_upstream_returns_empty_resolution() -> None:
    upstream = pl.DataFrame(
        schema={"id": pl.UInt64, "source": pl.Utf8, "key": pl.Utf8, "leaf": pl.UInt64}
    )
    resolution = materialise_resolution(_clusters([]), upstream)
    assert resolution.height == 0
    assert set(resolution.columns) == {"root", "leaf", "key", "source"}


def test_no_clusters_means_everything_falls_through() -> None:
    """A resolver whose models formed no edges keeps the upstream grouping intact."""
    upstream = _upstream(
        [
            (101, "A", "ka1", 1),
            (101, "A", "ka2", 2),
            (3, "A", "ka3", 3),
        ]
    )
    resolution = materialise_resolution(_clusters([]), upstream)
    assert _partition(resolution) == {
        frozenset({"ka1", "ka2"}),  # upstream cluster survives
        frozenset({"ka3"}),
    }


def test_the_adapter_mints_the_same_ids_a_resolver_does() -> None:
    """Scoring compares judged groups to resolved clusters *by ID*.

    If `_mint_cluster_id` and `root_id` ever diverge, every comparison misses and
    precision/recall is computed over an empty set — which is what happened when
    `root_id` was vectorised and this coupling was implicit.
    """
    from matchlab.adapters.duckdb import _mint_cluster_id  # noqa: PLC0415

    for leaves in ([5, 9], [1, 2, 3], [10**18, 7]):
        assert _mint_cluster_id(leaves) == root_id_of(leaves)

    # A singleton is deliberately the leaf itself, for evaluation's fallback.
    assert _mint_cluster_id([42]) == 42
