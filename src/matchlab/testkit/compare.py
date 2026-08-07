"""Scoring: turn results into the vocabulary, then compare them with the answer.

The rule for where comparison lives: **free functions convert and compare; methods exist
only to supply truth.** So the conversions and `diff_entities` are here, while
`LinkedSources.diff_resolver_output` and `.diff_model_edges` are methods — that object
is the only one that knows the planted answer.

Every comparison returns `(identical, report)`, where the report classifies each cluster
as perfect, subset, superset, wrong or invalid. That breakdown is the point: "eight
clusters were subsets" tells you the matcher is too strict, which a number cannot.
"""

from collections import Counter

import polars as pl

from matchlab.core.dsu import DisjointSet
from matchlab.testkit.entities import Cluster, EntityReference


def resolver_output_to_clusters(resolver_output: pl.DataFrame) -> set[Cluster]:
    """Convert a collected resolver output into entities comparable with truth.

    This is the other half of measuring a plan against generated data: the testkit
    plants known entities, the plan resolves records into clusters, and this turns
    those clusters back into the same currency the truth is expressed in. Pair it with
    [`LinkedSources.true_entity_subset()`][matchlab.testkit.linked.LinkedSources.true_entity_subset]
    and [`diff_entities()`][matchlab.testkit.compare.diff_entities]:

        identical, report = diff_entities(
            expected=linked.true_entity_subset("crn", "cdms"),
            actual=list(resolver_output_to_clusters(resolver_output)),
        )

    A `Cluster` compares by its keys and never by its ID
    (`Cluster.__eq__`), which is what lets this work at all: the resolver output's
    `root` is a content-derived hash minted at collect time and has no counterpart in
    the testkit's synthetic ID space. Only the `(source, key)` membership is comparable,
    and that is exactly what a cluster asserts.

    Args:
        resolver_output: A frame conforming to `SCHEMA_RESOLVER_OUTPUT` — the frame
            returned by `Resolver.entities()`, with `root`, `key` and `source` columns.

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
    """Convert scores to Cluster objects based on a threshold."""
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
    """Compare two lists of Cluster with detailed diff information.

    Args:
        expected: Expected Cluster list
        actual: Actual Cluster list

    Returns:
        A tuple containing:
        - Boolean: True if lists are identical, False otherwise
        - Dictionary that counts the number of actual entities that fall into the
            following criteria:
            - 'perfect': Match an expected entity exactly
            - 'subset': Are a subset of an expected entity
            - 'superset': Are a superset of an expected entity
            - 'wrong': Don't match any expected entity
            - 'invalid': Contain keys not present in any expected entity
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
