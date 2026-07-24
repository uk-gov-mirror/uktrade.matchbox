"""Test deterministic behavior of dedupers."""

from collections.abc import Callable
from typing import Any
from unittest.mock import Mock, patch

import polars as pl
import pytest

from matchlab.cleaning import Cleaner
from matchlab.core.factories.entities import FeatureConfig
from matchlab.core.factories.sources import (
    SourceTestkit,
    SourceTestkitParameters,
    linked_sources_factory,
)
from matchlab.models import Model
from matchlab.models.dedupers.base import Deduper
from matchlab.models.dedupers.naive import NaiveDeduper, NaiveSettings

DeduperConfigurator = Callable[[SourceTestkit], dict[str, Any]]

# Methodology configuration adapters


def configure_naive_deduper(testkit: SourceTestkit) -> dict[str, Any]:
    """Configure settings for NaiveDeduper.

    Args:
        testkit: SourceTestkit object from linked_sources_factory

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
@patch.object(Cleaner, "_frame")
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

    source_parameters = SourceTestkitParameters(
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
        name="exact_deduper",
        model_class=Deduper,
        model_settings=configure_deduper(source_testkit),
        left=source_testkit.source.clean(),
    )
    results = deduper.collect().edges()

    # Validate results against ground truth
    identical, report = linked.diff_model_edges(
        scores=results,
        left_clusters=source_testkit.entities,
        right_clusters=None,
        sources=["source_exact"],
        threshold=0,
    )

    assert identical, f"Expected perfect results but got: {report}"


@pytest.mark.parametrize(("Deduper", "configure_deduper"), DEDUPERS)
@patch.object(Cleaner, "_frame")
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

    source_parameters = SourceTestkitParameters(
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
        name="exact_deduper",
        model_class=Deduper,
        model_settings=configure_deduper(source),
        left=source.source.clean(),
    )
    results = deduper.collect().edges()

    # Validate results against ground truth
    identical, report = linked.diff_model_edges(
        scores=results,
        left_clusters=source.entities,
        right_clusters=None,
        sources=["source_exact"],
        threshold=0,
    )

    assert identical, f"Expected perfect results but got: {report}"
