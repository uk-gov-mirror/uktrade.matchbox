"""Linkers checked against scores, not just yes or no matches.

`SCORED_LINKERS` covers only linkers that emit a real probability rather than a fixed
0 or 1, so thresholding their scores can separate a subset of matches from the full
set.
"""

from collections.abc import Callable
from typing import Any
from unittest.mock import Mock, patch

import polars as pl
import pytest
from splink import SettingsCreator
from splink import comparison_library as cl

from matchlab.models import Model
from matchlab.models.linkers.base import Linker
from matchlab.models.linkers.splinklinker import SplinkLinker, SplinkSettings
from matchlab.models.linkers.weighteddeterministic import (
    WeightedDeterministicLinker,
    WeightedDeterministicSettings,
)
from matchlab.testkit.features import (
    FeatureConfig,
    ReplaceRule,
    SourceParameters,
    SuffixRule,
)
from matchlab.testkit.linked import linked_sources_factory
from matchlab.testkit.sources import GeneratedSource
from matchlab.views import View

LinkerConfigurator = Callable[[GeneratedSource, GeneratedSource], dict[str, Any]]

# Methodology configuration adapters


def configure_weighted_scored(
    left_testkit: GeneratedSource, right_testkit: GeneratedSource
) -> dict[str, Any]:
    """Build validated `WeightedDeterministicSettings` with geometric per-field weights.

    Each field's weight is half the previous one's, so no two fields matter equally and
    the resulting scores spread out instead of collapsing to a single value.
    """
    left_fields = {
        name for name in left_testkit.field_names if name not in ("key", "id")
    }
    right_fields = {
        name for name in right_testkit.field_names if name not in ("key", "id")
    }
    shared_fields: list[str] = sorted(left_fields & right_fields)

    if not shared_fields:
        raise ValueError("Must have at least one shared field")

    weights = [1 * (0.5**i) for i in range(len(shared_fields))]
    total_weight = sum(weights)

    # Weights must sum to 1, since the linker treats the total as a match probability.
    normalised_weights = [w / total_weight for w in weights]

    weighted_comparisons = []
    for field, weight in zip(shared_fields, normalised_weights, strict=True):
        weighted_comparisons.append(
            {"comparison": f"l.{field} = r.{field}", "weight": weight}
        )

    settings_dict = {
        "left_id": "id",
        "right_id": "id",
        "weighted_comparisons": weighted_comparisons,
        "threshold": 0.0,
    }

    WeightedDeterministicSettings.model_validate(settings_dict)

    return settings_dict


def configure_splink_scored(
    left_testkit: GeneratedSource, right_testkit: GeneratedSource
) -> dict[str, Any]:
    """Build validated `SplinkSettings`, choosing comparisons by each field's type."""
    left_fields = {
        name for name in left_testkit.field_names if name not in ("key", "id")
    }
    right_fields = {
        name for name in right_testkit.field_names if name not in ("key", "id")
    }
    shared_fields: list[str] = sorted(left_fields & right_fields)

    comparisons = []
    blocking_rules = []
    deterministic_matching_rules = []

    for field in shared_fields:
        field_type = next(
            (f.datatype for f in left_testkit.features if f.name == field),
            None,
        )

        deterministic_matching_rules.append(f"l.{field} = r.{field}")

        if field_type == pl.String:
            blocking_rules.append(f"SUBSTR(l.{field}, 1, 3) = SUBSTR(r.{field}, 1, 3)")
            comparisons.append(cl.JaroWinklerAtThresholds(field, [0.9, 0.7]))

        elif field_type in (pl.Int64, pl.Float64, pl.Decimal):
            blocking_rules.append(f"CAST(l.{field} AS INT) = CAST(r.{field} AS INT)")
            comparisons.append(cl.ExactMatch(field))

        else:
            comparisons.append(cl.ExactMatch(field))

    linker_settings = SettingsCreator(
        link_type="link_only",
        blocking_rules_to_generate_predictions=blocking_rules,
        comparisons=comparisons,
    )

    training_functions = [
        {
            "function": "estimate_probability_two_random_records_match",
            "arguments": {
                "deterministic_matching_rules": deterministic_matching_rules,
                "recall": 0.7,
            },
        },
        {
            "function": "estimate_u_using_random_sampling",
            "arguments": {"max_pairs": 1e4},
        },
    ]

    settings_dict = {
        "left_id": "id",
        "right_id": "id",
        "linker_training_functions": training_functions,
        "linker_settings": linker_settings,
        "threshold": 0.01,
    }

    SplinkSettings.model_validate(settings_dict)

    return settings_dict


SCORED_LINKERS = [
    pytest.param(SplinkLinker, configure_splink_scored, id="Splink"),
    pytest.param(
        WeightedDeterministicLinker,
        configure_weighted_scored,
        id="WeightedDeterministic",
    ),
]

# Test cases


@pytest.mark.parametrize(("Linker", "configure_linker"), SCORED_LINKERS)
@patch.object(View, "_read_cache")
def test_probabilistic_scores_vary(
    mock_query_run: Mock, Linker: Linker, configure_linker: LinkerConfigurator
) -> None:
    """Thresholding on the mean score keeps a strict subset of the full match set.

    Every match still appears when nothing is thresholded out. Raising the threshold to
    the mean score drops the weaker matches, but must never turn a real match wrong or
    invalid.
    """
    features = (
        FeatureConfig(
            name="company_name",
            base_generator="company",
            variations=[
                SuffixRule(suffix=" Ltd"),
                SuffixRule(suffix=" Limited"),
                ReplaceRule(old="&", new="and"),
            ],
            drop_base=False,
        ),
        FeatureConfig(
            name="id_number",
            base_generator="numerify",
            parameters=(("text", "######"),),
        ),
    )

    configs = (
        SourceParameters(name="source_left", features=features, n_true_entities=10),
        SourceParameters(name="source_right", features=features, n_true_entities=10),
    )

    linked = linked_sources_factory(source_parameters=configs, seed=42)
    for _testkit in linked.sources.values():
        _testkit.write_to_location()
    left_source = linked.sources["source_left"]
    right_source = linked.sources["source_right"]

    mock_query_run.side_effect = [
        pl.from_arrow(left_source.data),
        pl.from_arrow(right_source.data),
    ]

    linker = Model(
        model_class=Linker,
        model_settings=configure_linker(left_source, right_source),
        left=left_source.source.view(),
        right=right_source.source.view(),
    )

    results = linker.collect().edges()

    identical, report = linked.diff_model_edges(
        results, left=left_source, right=right_source
    )

    assert identical, f"Expected perfect results but got: {report}"

    identical, report = linked.diff_model_edges(
        results,
        left=left_source,
        right=right_source,
        threshold=results["score"].mean(),
    )

    assert not identical, f"Expected imperfect results but got: {report}"
    # Expect subsets of matches where connections weren't made
    assert report["subset"] > 0
    # Expect no wrong or invalid matches (perfect possible but unlikely)
    assert report["wrong"] == 0
    assert report["invalid"] == 0
