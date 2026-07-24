"""Tests for the Components resolver methodology."""

import polars as pl
import pytest

from matchlab.core.arrow import SCHEMA_CLUSTERS
from matchlab.resolvers import Components, ComponentsSettings


def test_components_compute_clusters_uses_thresholds() -> None:
    """Test thresholds are honoured by the Components.compute_clusters."""
    method = Components(settings=ComponentsSettings(thresholds={"model_a": 0.6}))
    model_edges = {
        "model_a": pl.DataFrame(
            {
                "left_id": [1, 2],
                "right_id": [2, 3],
                "score": [0.8, 0.4],
            },
            schema={
                "left_id": pl.UInt64,
                "right_id": pl.UInt64,
                "score": pl.Float32,
            },
        )
    }

    clusters = method.compute_clusters(model_edges=model_edges)

    grouped_clusters = {
        frozenset(group["child_id"].to_list())
        for group in clusters.partition_by("parent_id")
    }
    assert grouped_clusters == {frozenset({1, 2})}


def test_components_compute_clusters_returns_empty_for_no_edges() -> None:
    """Test Components.compute_clusters can work with no data."""
    clusters = Components(settings=ComponentsSettings()).compute_clusters(
        model_edges={}
    )
    assert clusters.height == 0
    assert clusters.schema == pl.Schema(SCHEMA_CLUSTERS)


def test_components_compute_clusters_merges_multiple_models() -> None:
    """Test Components.compute_clusters can work with multiple models."""
    method = Components(
        settings=ComponentsSettings(
            thresholds={"model_a": 0.0}  # one threshold implicit
        )
    )
    model_edges = {
        "model_a": pl.DataFrame(
            {"left_id": [1], "right_id": [2], "score": [0.9]},
            schema={
                "left_id": pl.UInt64,
                "right_id": pl.UInt64,
                "score": pl.Float32,
            },
        ),
        "model_b": pl.DataFrame(
            {"left_id": [3], "right_id": [4], "score": [0.8]},
            schema={
                "left_id": pl.UInt64,
                "right_id": pl.UInt64,
                "score": pl.Float32,
            },
        ),
    }

    clusters = method.compute_clusters(model_edges=model_edges)

    grouped_clusters = {
        frozenset(group["child_id"].to_list())
        for group in clusters.partition_by("parent_id")
    }
    assert grouped_clusters == {frozenset({1, 2}), frozenset({3, 4})}


def test_components_compute_clusters_errors_on_unknown_model() -> None:
    """Test that a threshold for an unknown model raises a RuntimeError."""
    method = Components(settings=ComponentsSettings(thresholds={"fooo": 0.5}))

    with pytest.raises(RuntimeError, match="Unknown models in thresholds"):
        method.compute_clusters(model_edges={"foo": pl.DataFrame()})
