"""Tests for the Components resolver methodology."""

import polars as pl

from matchlab.resolvers import Components, ComponentsSettings
from matchlab.resolvers.base import SCHEMA_CLUSTERS


def test_components_uses_thresholds() -> None:
    """Test thresholds are honoured by the Components.compute_clusters."""
    method = Components(settings=ComponentsSettings(thresholds={0: 0.6}))
    model_edges = {
        0: pl.DataFrame(
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


def test_components_no_edges() -> None:
    """Test Components.compute_clusters can work with no data."""
    clusters = Components(settings=ComponentsSettings()).compute_clusters(
        model_edges={}
    )
    assert clusters.height == 0
    assert clusters.schema == pl.Schema(SCHEMA_CLUSTERS)


def test_components_merges_models() -> None:
    """Test Components.compute_clusters can work with multiple models."""
    method = Components(
        settings=ComponentsSettings(
            thresholds={0: 0.0}  # the second input's threshold is implicit
        )
    )
    model_edges = {
        0: pl.DataFrame(
            {"left_id": [1], "right_id": [2], "score": [0.9]},
            schema={
                "left_id": pl.UInt64,
                "right_id": pl.UInt64,
                "score": pl.Float32,
            },
        ),
        1: pl.DataFrame(
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


def test_components_threshold_no_edges() -> None:
    """A methodology takes positions on trust; `Resolver` is what checks them.

    A threshold naming a model that is not an input used to be caught here, at collect
    time. It is now caught when the resolver is constructed, where the model object is
    still in hand — see `test_a_threshold_must_name_an_input`.
    """
    method = Components(settings=ComponentsSettings(thresholds={7: 0.5}))

    clusters = method.compute_clusters(
        model_edges={
            0: pl.DataFrame(
                {"left_id": [1], "right_id": [2], "score": [0.9]},
                schema={
                    "left_id": pl.UInt64,
                    "right_id": pl.UInt64,
                    "score": pl.Float32,
                },
            )
        }
    )

    assert {
        frozenset(group["child_id"].to_list())
        for group in clusters.partition_by("parent_id")
    } == {frozenset({1, 2})}
