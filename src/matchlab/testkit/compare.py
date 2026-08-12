"""Scoring: turn results into the vocabulary, then compare them with the answer.

The functions here convert a resolver's or model's output into `Cluster` objects, and
diff two `Cluster` lists. None of that needs the planted answer, which is why they stay
as plain functions rather than methods. `LinkedSources.diff_resolver_output()` and
`.diff_model_edges()` do need the planted answer, so those are methods on the one
object that holds it.

Every comparison returns `(identical, report)`, where the report classifies each cluster
as perfect, subset, superset, wrong or invalid. That breakdown is the point. "Eight
clusters were subsets" tells you the matcher is too strict, in a way a number cannot.
"""

from collections import Counter

import polars as pl

from matchlab.core.dsu import DisjointSet
from matchlab.testkit.entities import Cluster, EntityReference


def resolver_output_to_clusters(resolver_output: pl.DataFrame) -> set[Cluster]:
    """Convert a collected resolver output into entities comparable with truth.

    This is the other half of measuring a plan against generated data. The testkit
    plants known entities, the plan resolves records into clusters, and this turns
    those clusters back into the same currency the truth is expressed in. Pair it with
    [`LinkedSources.true_entity_subset()`][matchlab.testkit.linked.LinkedSources.true_entity_subset]
    and [`diff_entities()`][matchlab.testkit.compare.diff_entities]:

        identical, report = diff_entities(
            expected=linked.true_entity_subset("crn", "cdms"),
            actual=list(resolver_output_to_clusters(resolver_output)),
        )

    A `Cluster` compares by its keys and never by its ID (`Cluster.__eq__`), which is
    what lets this work at all. The resolver output's `root` is a content-derived hash
    minted at collect time, and has no counterpart in the testkit's synthetic ID space.
    Only the `(source, key)` membership is comparable, and that is exactly what a
    cluster asserts.

    Args:
        resolver_output: A table conforming to `SCHEMA_RESOLVER_OUTPUT`, the table
            `Resolver.entities()` returns, with `root`, `key` and `source` columns.

    Returns:
        One Cluster per distinct `root`.

    Raises:
        ValueError: If the required columns are absent.
    """
    required = {"root", "key", "source"}
    missing = required - set(resolver_output.columns)
    if missing:
        raise ValueError(
            f"Fields {sorted(missing)} must be included in the resolver output and "
            f"are missing. Available: {sorted(resolver_output.columns)}"
        )

    clusters = resolver_output.group_by("root").agg("source", "key")

    entities: set[Cluster] = set()
    for cluster in clusters.iter_rows(named=True):
        keys: dict[str, set[str]] = {}
        for source, key in zip(cluster["source"], cluster["key"], strict=True):
            keys.setdefault(source, set()).add(str(key))
        entities.add(
            Cluster(
                keys=EntityReference(
                    {source: frozenset(group) for source, group in keys.items()}
                )
            )
        )

    return entities


def scores_to_clusters(
    scores: pl.DataFrame,
    left_clusters: tuple[Cluster, ...],
    right_clusters: tuple[Cluster, ...] | None = None,
    threshold: float = 0.0,
) -> tuple[Cluster, ...]:
    """Merge clusters connected by a score at or above threshold.

    Left and right clusters that share an edge scoring at least `threshold` merge into
    one `Cluster`. With no `right_clusters`, this merges within `left_clusters` alone
    (deduplication).

    Args:
        scores: A `left_id`/`right_id`/`score` edge table.
        left_clusters: Clusters the model read as its left input.
        right_clusters: Clusters read as the right input, for a linker. `None` for a
            dedupe.
        threshold: Score at or above which an edge counts as a match.

    Returns:
        One merged `Cluster` per connected component.
    """
    left_lookup = {entity.id: entity for entity in left_clusters}
    if right_clusters is not None:
        right_lookup = {entity.id: entity for entity in right_clusters}
    else:
        right_lookup = left_lookup

    djs = DisjointSet[Cluster]()

    # Add ALL entities to the disjoint set
    for entity in left_clusters:
        djs.add(entity)
    if right_clusters is not None:
        for entity in right_clusters:
            djs.add(entity)

    # Add edges to the disjoint set
    for record in scores.to_dicts():
        if record["score"] >= threshold:
            djs.union(
                left_lookup[record["left_id"]],
                right_lookup[record["right_id"]],
            )

    components: set[set[Cluster]] = djs.get_components()

    entities: list[Cluster] = []
    for component in components:
        merged: Cluster = sum(component)
        entities.append(merged)

    return tuple(entities)


def diff_entities(expected: list[Cluster], actual: list[Cluster]) -> tuple[bool, dict]:
    """Compare two lists of Cluster against each other, with a diff report.

    Args:
        expected: The expected Cluster list.
        actual: The actual Cluster list.

    Returns:
        `(identical, report)`. `identical` is `True` if the two lists match exactly.
        `report` counts how each actual cluster relates to the expected ones:

        - `perfect`: matches an expected cluster exactly.
        - `subset`: is a subset of an expected cluster.
        - `superset`: is a superset of an expected cluster.
        - `wrong`: does not overlap any expected cluster.
        - `invalid`: contains keys absent from every expected cluster.
    """
    expected_set, actual_set = set(expected), set(actual)
    if expected_set == actual_set:
        return True, {}

    all_expected = sum(expected_set)
    perfect_matches = expected_set & actual_set
    remaining_actual = actual_set - perfect_matches

    counter = Counter(
        {
            "perfect": len(perfect_matches),
            "subset": 0,
            "superset": 0,
            "wrong": 0,
            "invalid": 0,
        }
    )

    for a in remaining_actual:
        if any(a in e for e in expected_set):
            counter["subset"] += 1
        elif a not in all_expected:
            counter["invalid"] += 1
        elif any(e in a for e in expected_set):
            counter["superset"] += 1
        else:
            counter["wrong"] += 1

    return False, dict(counter)
