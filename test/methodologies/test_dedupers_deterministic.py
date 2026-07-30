"""Test deterministic behavior of dedupers."""

from collections.abc import Callable
from typing import Any
from unittest.mock import Mock, patch

import polars as pl
import pytest

from matchlab.models import Model
from matchlab.models.dedupers.base import Deduper
from matchlab.models.dedupers.naive import NaiveDeduper, NaiveSettings
from matchlab.testkit.features import FeatureConfig, SourceParameters
from matchlab.testkit.linked import linked_sources_factory
from matchlab.testkit.sources import GeneratedSource
from matchlab.views import View

DeduperConfigurator = Callable[[GeneratedSource], dict[str, Any]]

# Methodology configuration adapters


def configure_naive_deduper(testkit: GeneratedSource) -> dict[str, Any]:
    """Configure settings for NaiveDeduper.

    Args:
        testkit: GeneratedSource object from linked_sources_factory

    Returns:
        A dictionary with validated settings for NaiveDeduper
    """
    # Extract field names excluding key and id
    fields = [name for name in testkit.field_names if name not in ("key", "id")]

    settings_dict = {
        "id": "id",
        "unique_fields": fields,
    }

    NaiveSettings.model_validate(settings_dict)

    return settings_dict


DEDUPERS = [
    pytest.param(NaiveDeduper, configure_naive_deduper, id="Naive"),
    # Add more deduper classes and configuration functions here
]

# Test cases


@pytest.mark.parametrize(("Deduper", "configure_deduper"), DEDUPERS)
@patch.object(View, "_read_cache")
def test_no_deduplication(
    mock_query_run: Mock,
    Deduper: Deduper,
    configure_deduper: DeduperConfigurator,
) -> None:
    """Test deduplication where there aren't actually any duplicates."""
    # Create a source with exact duplicates
    features = (
        FeatureConfig(
            name="company",
            base_generator="company",
        ),
        FeatureConfig(
            name="email",
            base_generator="email",
        ),
    )

    source_parameters = SourceParameters(
        name="source_exact",
        features=features,
        n_true_entities=10,
        repetition=0,  # Each entity appears once
    )

    linked = linked_sources_factory(source_parameters=(source_parameters,), seed=42)
    for _testkit in linked.sources.values():
        _testkit.write_to_location()
    source_testkit = linked.sources["source_exact"]

    mock_query_run.return_value = pl.from_arrow(source_testkit.data)

    # Mock query to server

    # Configure and run the deduper
    deduper = Model(
        model_class=Deduper,
        model_settings=configure_deduper(source_testkit),
        left=source_testkit.source.view(),
    )
    results = deduper.collect().edges()

    # Validate results against ground truth
    identical, report = linked.diff_model_edges(results, left=source_testkit)

    assert identical, f"Expected perfect results but got: {report}"


@pytest.mark.parametrize(("Deduper", "configure_deduper"), DEDUPERS)
@patch.object(View, "_read_cache")
def test_exact_duplicate_deduplication(
    mock_query_run: Mock, Deduper: Deduper, configure_deduper: DeduperConfigurator
) -> None:
    """Test deduplication with exact duplicates."""
    # Create a source with exact duplicates
    features = (
        FeatureConfig(
            name="company",
            base_generator="company",
        ),
        FeatureConfig(
            name="email",
            base_generator="email",
        ),
    )

    source_parameters = SourceParameters(
        name="source_exact",
        features=features,
        n_true_entities=10,
        repetition=2,  # Each entity appears 3 times (base + 2 repetitions)
    )

    linked = linked_sources_factory(source_parameters=(source_parameters,), seed=42)
    for _testkit in linked.sources.values():
        _testkit.write_to_location()
    source = linked.sources["source_exact"]

    mock_query_run.return_value = pl.from_arrow(source.data)

    # Configure and run the deduper
    deduper = Model(
        model_class=Deduper,
        model_settings=configure_deduper(source),
        left=source.source.view(),
    )
    results = deduper.collect().edges()

    # Validate results against ground truth
    identical, report = linked.diff_model_edges(results, left=source)

    assert identical, f"Expected perfect results but got: {report}"
