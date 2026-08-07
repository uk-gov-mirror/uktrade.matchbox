"""Score a resolver's clusters against stored human judgements."""

from itertools import chain, combinations
from typing import TypeAlias

import polars as pl

Pair: TypeAlias = tuple[int, int]
Pairs: TypeAlias = set[Pair]

PrecisionRecall: TypeAlias = tuple[float, float]


def precision_recall(
    models_root_leaf: list[pl.DataFrame],
    judgements: pl.DataFrame,
    expansion: pl.DataFrame,
) -> list[PrecisionRecall]:
    """Compute precision and recall for each model's clusters against judgements.

    The function turns both model clusters and judgements into pairs of leaves, then
    compares them:

    - Convert each model's clusters and the judgements into pairwise connections
      between leaves. For judgements, this includes pairs the user was shown but
      rejected, not just the ones they endorsed. Each pair's net score is the number
      of times it was endorsed, minus the number of times it was rejected.
    - Keep only pairs whose leaves appear in every model and in the judgements, so
      each model is compared over the same data.
    - Drop a pair from both model and validation pairs if it was rejected as often as
      it was endorsed. If it was rejected more often than endorsed, drop it from
      validation pairs only, and keep it in model pairs.
    - Compute precision and recall for each model against the remaining validation
      pairs.

    This ignores user IDs, so judgements are not yet weighted by reviewer.

    Args:
        models_root_leaf: One `(root, leaf)` table per model. Each must be
            merge-forward, including every leaf reachable from the model's inputs,
            even ones no model touched, which appear as their own root.
        judgements: Table following `SCHEMA_JUDGEMENTS`.
        expansion: Table following `SCHEMA_CLUSTER_EXPANSION`.

    Returns:
        One `(precision, recall)` pair per model, in the order given.
    """
    leaves_per_set: list[set[int]] = []  # one entry for each model, one for judgements
    pairs_per_model: list[Pairs] = []

    if not len(judgements):
        raise ValueError("Judgements data cannot be empty.")

    # Process models and judgements
    for root_leaf in models_root_leaf:
        if not len(root_leaf):
            raise ValueError("Model data cannot be empty.")
        leaves_per_set.append(set(root_leaf["leaf"].to_list()))
        clusters = (
            root_leaf.group_by("root")
            .agg(pl.col("leaf").alias("leaves"))
            .select("leaves")
            .to_series()
            .to_list()
        )
        model_pairs = set()
        for c in clusters:
            model_pairs.update(combinations(sorted(c), r=2))
        pairs_per_model.append(model_pairs)

    validation_pairs, validation_net_count, validation_leaves = process_judgements(
        judgements, expansion
    )
    leaves_per_set.append(validation_leaves)

    # Filter pairs to the leaves shared by every model and the judgements. For
    # example, if model1 has (1,2), (1,3), (2,3), model2 has (1,10), (2,20), and
    # judgements have (1), (2,3), only (1,2) is kept.
    shared_leaves = set.intersection(*leaves_per_set)

    for i, model_pairs in enumerate(pairs_per_model):
        pairs_per_model[i] = {
            (a, b)
            for a, b in model_pairs
            if (
                a in shared_leaves
                and b in shared_leaves
                # Remove neutrally-judged pairs from the model
                and validation_net_count[(a, b)] != 0
            )
        }

    validation_pairs = {
        (a, b)
        for a, b in validation_pairs
        if a in shared_leaves
        and b in shared_leaves
        # Remove all neutrally or negatively judged pairs
        and validation_net_count[(a, b)] > 0
    }

    if not validation_pairs:
        raise ValueError("Validation data has no pairs to evaluate.")

    # Compute PR scores for each model
    pr_scores: list[PrecisionRecall] = []
    for i, model_pairs in enumerate(pairs_per_model):
        true_positive_pairs = model_pairs & validation_pairs
        if not model_pairs:
            raise ValueError(f"Model at index {i} has no pairs to evaluate.")
        pr_scores.append(
            (
                len(true_positive_pairs) / len(model_pairs),
                len(true_positive_pairs) / len(validation_pairs),
            )
        )

    return pr_scores


def process_judgements(
    judgements: pl.DataFrame, expansion: pl.DataFrame
) -> tuple[Pairs, dict[Pair, float], set[int]]:
    """Turn judgements into leaf pairs, a net score per pair, and the leaves shown.

    Expanding a cluster's leaves into every sorted pair is not quite enough on its
    own. When a user is shown (123) and endorses (12), they also reject pairs (1,3)
    and (2,3), not just endorse (1,2). Each pair's net score sums +1 for every
    endorsement and -1 for every rejection implied this way.

    This function assumes well-formed input:

    - Every shown cluster ID has a matching row in `expansion`.
    - Every endorsed cluster ID has a matching row in `expansion`, unless it is a
      single leaf.
    - No cluster is split across judgement rows unless every other part is expanded
      too. For example, if (123)->(12) appears, (123)->(3) must appear too.

    Args:
        judgements: Table following `SCHEMA_JUDGEMENTS`.
        expansion: Table following `SCHEMA_CLUSTER_EXPANSION`.

    Returns:
        A tuple of:

        - Every pair implied by an endorsement or a rejection.
        - Each pair's net score, where positive means endorsed more than rejected.
        - Every leaf ID shown to a user.
    """
    expanded_judgements = (
        judgements.join(expansion, left_on="shown", right_on="root")
        .rename({"leaves": "shown_leaves"})
        .join(
            expansion, left_on="endorsed", right_on="root", how="left"
        )  # left join as singleton leaves won't be expanded
        .rename({"leaves": "endorsed_leaves"})
        # Assume a missing expansion means a singleton leaf
        .with_columns(
            pl.when(pl.col("endorsed_leaves").is_null())
            .then(
                pl.col("endorsed").map_elements(
                    lambda x: [x], return_dtype=pl.List(pl.UInt64)
                )
            )
            .otherwise(pl.col("endorsed_leaves"))
            .alias("endorsed_leaves")
        )
    )

    all_leaves = set(
        chain.from_iterable(expanded_judgements["endorsed_leaves"].to_list())
    )

    if len(expanded_judgements) != len(judgements):
        raise ValueError("Malformed judgements / expansion data received.")

    validation_net_count: dict[Pair, float] = {}

    for row in expanded_judgements.rows(named=True):
        shown = row["shown_leaves"]
        endorsed = row["endorsed_leaves"]

        positive_pairs = set(combinations(sorted(endorsed), r=2))
        potential_pairs = set(combinations(sorted(shown), r=2))
        negative_pairs = potential_pairs - positive_pairs

        # -- WORKED EXAMPLE --
        # A user is shown cluster (1234) and endorses three groups: (1), (23), (4).
        # This makes pair (2,3) good, and (1,2), (1,3), (1,4), (2,4), (3,4) bad.
        # Target result: +1 for (2,3), -1 for every bad pair.

        # The data does not arrive as one row holding the whole judgement. It arrives
        # as one row per endorsed group, and these rows can be mixed with other users'
        # judgements:
        #   Row A: shown (1234), endorsed (1)
        #   Row B: shown (1234), endorsed (23)
        #   Row C: shown (1234), endorsed (4)
        # Each row below is scored on its own. The partial scores still add up to the
        # right total, whatever order the rows arrive in.

        # Row A: shown (1234), endorsed (1)
        # - Potential pairs: (1,2) (1,3) (1,4) (2,3) (2,4) (3,4).
        # - Endorsed pairs: none. A single item forms no pair.
        # - Every potential pair is tentatively rejected, weight -1/4, because this
        #   endorsed group holds 1 of the 4 shown items.
        # - Running scores: all pairs = -0.25.

        # Row B: shown (1234), endorsed (23)
        # - Endorsed pair (2,3) gets +1, plus +2/4 to cancel the rejection weight it
        #   took in rows A and C, the non-endorsed share of this group.
        #   Total for (2,3): +1 + 0.5 = +1.5.
        # - Remaining rejected pairs take a further -2/4 = -0.5.
        # - Running scores: (2,3) = -0.25 + 1.5 = +1.25, others = -0.25 - 0.5 = -0.75.

        # Row C: shown (1234), endorsed (4)
        # - Final -1/4 rejection weight lands on every potential pair.
        # - Final scores: (2,3) = +1.25 - 0.25 = +1.0, others = -0.75 - 0.25 = -1.0.

        # Result: (2,3) nets +1, matching the endorsement, and every rejected pair
        # nets -1, whatever order the rows arrive in or however they interleave with
        # other judgements, as long as no cluster is split without expanding its
        # other part too.

        # Subtract negative weight for all shown pairs not endorsed on this row
        negative_adjustment = len(endorsed) / len(shown)
        validation_net_count.update(
            {
                p: validation_net_count.get(p, 0) - negative_adjustment
                for p in negative_pairs
            }
        )

        # Add 1 plus positive weight for all pairs endorsed on this row
        positive_adjustment = 1 + ((len(shown) - len(endorsed)) / len(shown))
        validation_net_count.update(
            {
                p: validation_net_count.get(p, 0) + positive_adjustment
                for p in positive_pairs
            }
        )

    all_pairs = set(validation_net_count.keys())

    return all_pairs, validation_net_count, all_leaves
