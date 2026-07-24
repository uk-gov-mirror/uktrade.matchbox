"""Phase 0 spike: materialize-forward vs. server resolve-on-demand.

The `matchlab` plan replaces the server's on-demand resolution engine
(`server/postgresql/utils/query.py`: `_build_unified_query`, which projects source
keys up the resolver hierarchy with a priority-`COALESCE` at *query time*) with
*materialize-forward*: each resolver, when collected, stores its **complete** flat
resolution `(source, key, leaf, root)` to disk, and downstream steps / terminal ops
(`get_matches`, `lookup_key`) just read it.

This spike answers the one semantic question that decides whether that swap is safe:

    Does materialize-forward reproduce the server's resolve-on-demand result for
    *layered* resolvers — including leaves that an upstream resolver grouped but a
    downstream resolver never touched ("fall-through" leaves)?

It is deliberately infra-free (no Postgres/redis/S3). Instead it encodes a faithful
*reference* of the server's `COALESCE`-priority semantics (derived directly from
`_build_unified_query` + `Steps.get_lineage`, orm.py:326 "ordered by priority, highest
priority / lowest level first") and compares three client-side strategies against it.

Key finding (see `test_fallthrough_*`): the naive "each resolver stores only its own
clusters" is WRONG. A collected resolver must store `merge(upstream complete
resolution, its own clusters)` — its clusters override upstream for touched leaves, and
untouched leaves inherit their upstream root. That merge is the eager equivalent of the
server's `COALESCE`. This is the design constraint the spike exists to surface, and it
feeds straight back into the Phase 1 `Adapter.store_resolver` contract.

Run:
    uv run python spikes/phase0_materialize_forward.py     # prints a report
    uv run pytest spikes/phase0_materialize_forward.py     # asserts equivalence
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pyarrow as pa

# The only import from the package: the repo's own relabel-invariant cluster hash,
# which we reuse as the equivalence oracle. It lives in matchlab.core.hash, which is
# pure (no client settings, no server), so importing it does not trigger the
# client-settings coupling this whole project is removing.
from matchlab.core.hash import hash_clusters

Leaf = str
Root = str

# --------------------------------------------------------------------------------------
# Scenario model
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Resolver:
    """A resolver step, modelled at leaf granularity (as the server stores it).

    `clusters` are groups of *leaves* this resolver merges. Following the server, the
    leaves are already **expanded**: when a resolver operates on upstream-resolved data
    and merges an upstream root, the server's `Contains` table records every leaf under
    that root, so we list those leaves here too.

    `level` mirrors `StepFrom.level`: 0 == apex == highest query priority. Lower level
    wins in the server's `COALESCE`.
    """

    name: str
    level: int
    clusters: tuple[frozenset[Leaf], ...]

    def root_of(self, cluster: frozenset[Leaf]) -> Root:
        """Deterministic, content-derived root label for a cluster of leaves.

        Real code hashes the leaf-set; here the sorted tuple keeps the report readable.
        """
        return f"{self.name}:{'+'.join(sorted(cluster))}"

    def leaf_to_root(self) -> dict[Leaf, Root]:
        """This resolver's own opinion: leaf -> root, only for leaves it touches."""
        out: dict[Leaf, Root] = {}
        for cluster in self.clusters:
            root = self.root_of(cluster)
            for leaf in cluster:
                out[leaf] = root
        return out


@dataclass
class Scenario:
    """A layered DAG: source keys -> leaves, plus resolvers at various levels."""

    key_to_leaf: dict[tuple[str, str], Leaf]  # (source, key) -> leaf
    resolvers: list[Resolver] = field(default_factory=list)

    @property
    def all_leaves(self) -> list[Leaf]:
        """Every distinct leaf across all sources, sorted."""
        return sorted(set(self.key_to_leaf.values()))

    def resolvers_high_to_low(self) -> list[Resolver]:
        """Priority order used by the server's COALESCE: lowest level first."""
        return sorted(self.resolvers, key=lambda r: (r.level, r.name))

    def resolvers_low_to_high(self) -> list[Resolver]:
        """Collect order: most-upstream first, apex last."""
        return sorted(self.resolvers, key=lambda r: (-r.level, r.name))


# --------------------------------------------------------------------------------------
# (1) Reference: the server's resolve-on-demand semantics
# --------------------------------------------------------------------------------------


def resolve_on_demand(scenario: Scenario) -> dict[Leaf, Root]:
    """Faithful reference of `_build_unified_query`'s COALESCE-priority projection.

    For each leaf, walk the resolvers highest-priority-first and take the first one
    that has an opinion; fall back to the leaf itself (its source cluster). This is
    exactly `COALESCE(resolver_1_root, ..., resolver_n_root, leaf)` with resolver_1 the
    highest-priority (lowest-level) step.
    """
    priority = scenario.resolvers_high_to_low()
    opinions = [(r.name, r.leaf_to_root()) for r in priority]

    resolution: dict[Leaf, Root] = {}
    for leaf in scenario.all_leaves:
        root: Root = leaf
        for _name, leaf_to_root in opinions:  # highest priority first
            if leaf in leaf_to_root:
                root = leaf_to_root[leaf]
                break
        resolution[leaf] = root
    return resolution


# --------------------------------------------------------------------------------------
# (2) & (3): client-side materialize-forward strategies
# --------------------------------------------------------------------------------------


def materialize_forward_merge(scenario: Scenario) -> dict[Leaf, Root]:
    """CORRECT strategy: each resolver stores merge(upstream complete, own clusters).

    Process resolvers most-upstream first. Start from singletons (each leaf its own
    root); each resolver overrides the roots of the leaves it touches. Untouched leaves
    keep their upstream root. The apex's stored map is complete over all leaves — the
    eager equivalent of the server's COALESCE.
    """
    resolution: dict[Leaf, Root] = {leaf: leaf for leaf in scenario.all_leaves}
    for resolver in scenario.resolvers_low_to_high():  # upstream first, apex last
        for cluster in resolver.clusters:
            root = resolver.root_of(cluster)
            for leaf in cluster:
                resolution[leaf] = root
    return resolution


def materialize_forward_naive(scenario: Scenario) -> dict[Leaf, Root]:
    """WRONG strategy: the apex resolver stores only its OWN clusters.

    Leaves resolved only by an upstream resolver (fall-through) are lost and collapse
    back to singletons. Kept here to demonstrate the divergence the merge strategy
    fixes.
    """
    apex = min(scenario.resolvers, key=lambda r: (r.level, r.name))
    resolution: dict[Leaf, Root] = {leaf: leaf for leaf in scenario.all_leaves}
    for cluster in apex.clusters:
        root = apex.root_of(cluster)
        for leaf in cluster:
            resolution[leaf] = root
    return resolution


# --------------------------------------------------------------------------------------
# Equivalence oracle: compare partitions, invariant to root relabeling
# --------------------------------------------------------------------------------------


def partition(resolution: dict[Leaf, Root]) -> frozenset[frozenset[Leaf]]:
    """The set of leaf-groups, forgetting the (arbitrary) root labels."""
    groups: dict[Root, set[Leaf]] = {}
    for leaf, root in resolution.items():
        groups.setdefault(root, set()).add(leaf)
    return frozenset(frozenset(g) for g in groups.values())


def canonical_hash(resolution: dict[Leaf, Root]) -> bytes:
    """Relabel-invariant hash using the repo's own `hash_clusters`.

    Proves the eventual equivalence check reuses existing machinery rather than a
    bespoke comparator. `hash_clusters` is documented invariant to parent_id relabeling
    and child ordering, so two resolutions with the same partition hash equal.
    """
    leaf_id = {leaf: i for i, leaf in enumerate(sorted(resolution), start=1)}
    root_id: dict[Root, int] = {}

    def rid(root: Root) -> int:
        return root_id.setdefault(root, 10_000 + len(root_id))

    parents = [rid(root) for root in resolution.values()]
    children = [leaf_id[leaf] for leaf in resolution]
    table = pa.table(
        {
            "parent_id": pa.array(parents, pa.uint64()),
            "child_id": pa.array(children, pa.uint64()),
        }
    )
    return hash_clusters(table)


# --------------------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------------------


def layered_scenario_with_fallthrough() -> Scenario:
    """Sources A/B/C, per-source dedupe, one apex link that ignores B-extra and all C.

    * dedupe_A merges {a1, a2}; a3 stays alone.
    * dedupe_C merges {c1, c2}.
    * link (apex) merges {a1, a2, b1} (a1/a2 expanded from dedupe_A's root).
      It never touches a3, b2, or C.

    Fall-through leaves: c1, c2 (only dedupe_C has an opinion) and a3, b2.
    """
    key_to_leaf = {
        ("A", "a1"): "a1",
        ("A", "a2"): "a2",
        ("A", "a3"): "a3",
        ("B", "b1"): "b1",
        ("B", "b2"): "b2",
        ("C", "c1"): "c1",
        ("C", "c2"): "c2",
    }
    dedupe_a = Resolver("dedupe_A", level=1, clusters=(frozenset({"a1", "a2"}),))
    dedupe_c = Resolver("dedupe_C", level=1, clusters=(frozenset({"c1", "c2"}),))
    link = Resolver("link", level=0, clusters=(frozenset({"a1", "a2", "b1"}),))
    return Scenario(key_to_leaf, [dedupe_a, dedupe_c, link])


def simple_single_resolver_scenario() -> Scenario:
    """One dedupe resolver, no layering — the trivial control case."""
    key_to_leaf = {("A", k): k for k in ("a1", "a2", "a3")}
    dedupe = Resolver("dedupe_A", level=0, clusters=(frozenset({"a1", "a2"}),))
    return Scenario(key_to_leaf, [dedupe])


def three_layer_scenario() -> Scenario:
    """Dedupe -> link -> super-link, to check the merge composes past two layers."""
    key_to_leaf = {
        ("A", "a1"): "a1",
        ("A", "a2"): "a2",
        ("B", "b1"): "b1",
        ("C", "c1"): "c1",
        ("C", "c2"): "c2",
        ("D", "d1"): "d1",
    }
    dedupe_a = Resolver("dedupe_A", level=2, clusters=(frozenset({"a1", "a2"}),))
    dedupe_c = Resolver("dedupe_C", level=2, clusters=(frozenset({"c1", "c2"}),))
    link_ab = Resolver("link_AB", level=1, clusters=(frozenset({"a1", "a2", "b1"}),))
    # apex links the AB cluster with C, still ignoring D entirely
    super_link = Resolver(
        "super", level=0, clusters=(frozenset({"a1", "a2", "b1", "c1", "c2"}),)
    )
    return Scenario(key_to_leaf, [dedupe_a, dedupe_c, link_ab, super_link])


SCENARIOS = {
    "single": simple_single_resolver_scenario,
    "layered_fallthrough": layered_scenario_with_fallthrough,
    "three_layer": three_layer_scenario,
}


# --------------------------------------------------------------------------------------
# pytest
# --------------------------------------------------------------------------------------


def _fmt(partition_set: frozenset[frozenset[Leaf]]) -> str:
    """Render a partition as `{a,b} {c}` for the report."""
    return "  ".join(
        "{" + ",".join(sorted(g)) + "}"
        for g in sorted(partition_set, key=lambda g: sorted(g))
    )


def test_merge_matches_server_on_all_scenarios() -> None:
    """Merge-forward equals the server reference on every scenario."""
    for name, build in SCENARIOS.items():
        scenario = build()
        reference = resolve_on_demand(scenario)
        merged = materialize_forward_merge(scenario)
        assert partition(merged) == partition(reference), f"partition mismatch: {name}"
        assert canonical_hash(merged) == canonical_hash(reference), f"hash: {name}"


def test_naive_diverges_on_fallthrough() -> None:
    """The whole point: naive-forward is wrong exactly when leaves fall through."""
    scenario = layered_scenario_with_fallthrough()
    reference = resolve_on_demand(scenario)
    naive = materialize_forward_naive(scenario)
    assert partition(naive) != partition(reference)
    # And specifically, it splits the fall-through C cluster back into singletons.
    assert frozenset({"c1", "c2"}) in partition(reference)
    assert frozenset({"c1", "c2"}) not in partition(naive)


def test_naive_ok_without_layering() -> None:
    """Naive == merge when there is nothing upstream to fall through from."""
    scenario = simple_single_resolver_scenario()
    reference = resolve_on_demand(scenario)
    assert partition(materialize_forward_naive(scenario)) == partition(reference)


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------


def main() -> None:
    """Print a per-scenario comparison of the three strategies."""
    print("Phase 0 spike — materialize-forward vs. server resolve-on-demand\n")
    for name, build in SCENARIOS.items():
        scenario = build()
        reference = resolve_on_demand(scenario)
        merged = materialize_forward_merge(scenario)
        naive = materialize_forward_naive(scenario)

        ref_p, merge_p, naive_p = map(partition, (reference, merged, naive))
        merge_ok = ref_p == merge_p and canonical_hash(merged) == canonical_hash(
            reference
        )
        naive_ok = ref_p == naive_p

        print(f"── scenario: {name}")
        print(f"   server (reference) : {_fmt(ref_p)}")
        print(f"   merge-forward      : {_fmt(merge_p)}   {'OK' if merge_ok else 'X'}")
        print(f"   naive-forward      : {_fmt(naive_p)}   {'OK' if naive_ok else 'X'}")
        print()

    print(
        "Finding: merge-forward reproduces the server on every scenario; naive\n"
        "diverges as soon as a leaf falls through an untouched upstream resolver.\n"
        "=> store_resolver must persist merge(upstream complete, own clusters)."
    )


if __name__ == "__main__":
    main()
