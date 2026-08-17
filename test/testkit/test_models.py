"""Tests for generated models and their scored edges."""

from typing import Any, Literal

import polars as pl
import pytest

from matchlab.core.schemas import SCHEMA_MODEL_EDGES
from matchlab.sources import RelationalDB, Source
from matchlab.testkit.entities import Cluster
from matchlab.testkit.linked import linked_sources_factory
from matchlab.testkit.models import GeneratedModel


def _over_merged_scores(
    left: tuple[Cluster, ...], right: tuple[Cluster, ...] | None
) -> pl.DataFrame:
    """Edges from a deliberately wrong methodology that collapses everything into one.

    Stands in for a real methodology that over-matches. With more than one entity
    planted, the single resulting cluster must be reported as a *superset* of the
    expected ones, which is what makes this a check on the diff, not on chance.
    """
    left_ids = [entity.id for entity in left]
    if right is None:
        pairs = [(left_ids[0], other) for other in left_ids[1:]]
    else:
        right_ids = [entity.id for entity in right]
        pairs = [(left_ids[0], other) for other in right_ids] + [
            (other, right_ids[0]) for other in left_ids[1:]
        ]

    return pl.DataFrame(
        {
            "left_id": [left_id for left_id, _ in pairs],
            "right_id": [right_id for _, right_id in pairs],
            "score": [1.0] * len(pairs),
        },
        schema=pl.Schema(SCHEMA_MODEL_EDGES),
    )


def _build(how: str) -> GeneratedModel:
    """Build a generated model by each of the two supported routes."""
    linked = linked_sources_factory()
    if how == "dedupe":
        return linked.dedupe("crn", seed=13)
    return linked.link("crn", "cdms", seed=13)


@pytest.mark.parametrize(
    ("how", "expected_type", "should_have_right"),
    [
        pytest.param("dedupe", "deduper", False, id="dedupe_over_a_source"),
        pytest.param("linker", "linker", True, id="link_between_two_sources"),
    ],
)
def test_model_type_creation(
    how: str,
    expected_type: str,
    should_have_right: bool,
) -> None:
    """A generated model has the requested type, scored edges, and left source ids."""
    model = _build(how)

    assert model.model.model_type == expected_type
    assert (model.right_source is not None) == should_have_right

    assert len(model.scores) > 0
    assert model.scores.schema == pl.Schema(SCHEMA_MODEL_EDGES)

    assert "id" in model.left_source.data.column_names
    assert len(set(model.left_source.data["id"].to_pylist())) > 0

    if expected_type == "linker":
        left_ids = set(model.left_source.data["id"].to_pylist())
        right_ids = set(model.right_source.data["id"].to_pylist())
        assert not (left_ids & right_ids), (
            "Left and right IDs should be disjoint in linker"
        )

        score_left_ids = set(model.scores["left_id"].to_list())
        score_right_ids = set(model.scores["right_id"].to_list())
        assert score_left_ids <= left_ids, (
            "Probability left IDs should be subset of left IDs"
        )
        assert score_right_ids <= right_ids, (
            "Probability right IDs should be subset of right IDs"
        )


@pytest.mark.parametrize(
    ("left_testkit", "right_testkit", "model_type"),
    [
        pytest.param(
            "crn",
            None,
            "deduper",
            id="test_initial_deduper_methodology",
        ),
        pytest.param(
            "cdms",
            None,
            "deduper",
            id="test_second_deduper_methodology",
        ),
        pytest.param(
            "crn",
            "cdms",
            "linker",
            id="test_final_linker_methodology",
        ),
    ],
)
def test_model_pipeline_with_dummy_methodology(
    left_testkit: str,
    right_testkit: str | None,
    model_type: Literal["deduper", "linker"],
) -> None:
    """A methodology can be checked from one scores table, at any pipeline position.

    A perfect model recovers the planted entities exactly. Swapping in an
    over-matching methodology, one giant cluster, makes the diff report a superset
    instead, whatever position the model sits in.
    """
    linked = linked_sources_factory()

    # Create and validate perfect model
    left = linked.sources[left_testkit]
    right = linked.sources[right_testkit] if model_type == "linker" else None
    perfect_model = (
        linked.link(left_testkit, right_testkit)
        if right is not None
        else linked.dedupe(left_testkit)
    )

    identical, _ = linked.diff_model_edges(perfect_model.scores, left=left, right=right)
    assert identical, "A perfect model should recover the planted entities"

    identical, report = linked.diff_model_edges(
        _over_merged_scores(
            left.input_clusters, right.input_clusters if right else None
        ),
        left=left,
        right=right,
    )

    # Over-matching everything into a single cluster is a superset of each entity it
    # swallowed, and nothing matches perfectly.
    assert not identical
    assert report["superset"] > 0
    assert report["perfect"] == 0


@pytest.mark.parametrize(
    ("kwargs", "expected_error", "expected_message"),
    [
        pytest.param(
            {"model_type": "deduper", "score_range": (0.9, 0.8)},
            ValueError,
            "Scores must be increasing values between 0 and 1",
            id="invalid_score_range_decreasing",
        ),
        pytest.param(
            {"model_type": "deduper", "score_range": (-0.1, 0.8)},
            ValueError,
            "Scores must be increasing values between 0 and 1",
            id="invalid_score_range_negative",
        ),
        pytest.param(
            {"model_type": "deduper", "score_range": (0.8, 1.1)},
            ValueError,
            "Scores must be increasing values between 0 and 1",
            id="invalid_score_range_too_high",
        ),
    ],
)
def test_score_range_validation(
    kwargs: dict[str, Any], expected_error: type[Exception], expected_message: str
) -> None:
    """dedupe/link reject a score_range that isn't increasing and within 0 to 1."""
    linked = linked_sources_factory()
    with pytest.raises(expected_error, match=expected_message):
        linked.dedupe("crn", score_range=kwargs["score_range"])


@pytest.mark.parametrize(
    (
        "model_type",
        "n_true_entities",
        "score_range",
        "seed",
        "expected_checks",
    ),
    [
        pytest.param(
            "deduper",
            5,
            (0.8, 0.9),
            42,
            {
                "type": "deduper",
                "entity_count": 5,
                "has_right": False,
                "score_min": 0.8,
                "score_max": 0.9,
            },
            id="basic_deduper",
        ),
        pytest.param(
            "linker",
            10,
            (0.7, 0.8),
            42,
            {
                "type": "linker",
                "entity_count": 10,
                "has_right": True,
                "score_min": 0.7,
                "score_max": 0.8,
            },
            id="basic_linker",
        ),
        pytest.param(
            "deduper",
            100,
            (0.9, 1.0),
            42,
            {
                "type": "deduper",
                "entity_count": 100,
                "has_right": False,
                "score_min": 0.9,
                "score_max": 1.0,
            },
            id="large_deduper",
        ),
        pytest.param(
            "linker",
            20,
            (0.95, 1.0),
            42,
            {
                "type": "linker",
                "entity_count": 20,
                "has_right": True,
                "score_min": 0.95,
                "score_max": 1.0,
            },
            id="strict_linker",
        ),
    ],
)
def test_generated_model_basic_creation(
    model_type: str,
    n_true_entities: int,
    score_range: tuple[float, float],
    seed: int,
    expected_checks: dict,
) -> None:
    """A generated model's scores stay within the requested range, for either type."""
    linked = linked_sources_factory(n_true_entities=n_true_entities, seed=seed)
    model = (
        linked.link("crn", "cdms", score_range=score_range, seed=seed)
        if model_type == "linker"
        else linked.dedupe("crn", score_range=score_range, seed=seed)
    )

    assert str(model.model.model_type) == expected_checks["type"]

    assert (model.right_source is not None) == expected_checks["has_right"]
    assert len(model.predicted_clusters) == expected_checks["entity_count"]

    score_values = model.scores["score"].to_numpy()
    assert all(p >= expected_checks["score_min"] for p in score_values)
    assert all(p <= expected_checks["score_max"] for p in score_values)


@pytest.mark.parametrize(
    ("source_params", "expected_checks"),
    [
        pytest.param(
            {
                "left_name": "crn",
                "right_name": None,
                "true_entities_slice": slice(None),  # All entities
                "score_range": (0.8, 0.9),
            },
            {
                "type": "deduper",
                "has_right": False,
                "score_min": 0.8,
                "score_max": 0.9,
            },
            id="deduper_full_entities",
        ),
        pytest.param(
            {
                "left_name": "crn",
                "right_name": "cdms",
                "true_entities_slice": slice(None),
                "score_range": (0.8, 0.9),
            },
            {
                "type": "linker",
                "has_right": True,
                "score_min": 0.8,
                "score_max": 0.9,
            },
            id="linker_full_entities",
        ),
        pytest.param(
            {
                "left_name": "crn",
                "right_name": None,
                "true_entities_slice": slice(0, 1),  # Just first entity
                "score_range": (0.9, 1.0),
            },
            {
                "type": "deduper",
                "has_right": False,
                "score_min": 0.9,
                "score_max": 1.0,
            },
            id="deduper_partial_entities",
        ),
        pytest.param(
            {
                "left_name": "crn",
                "right_name": "cdms",
                "true_entities_slice": slice(0, 2),  # First two entities
                "score_range": (0.7, 0.8),
            },
            {
                "type": "linker",
                "has_right": True,
                "score_min": 0.7,
                "score_max": 0.8,
            },
            id="linker_partial_entities",
        ),
    ],
)
def test_generated_model_over_chosen_sources(
    source_params: dict, expected_checks: dict
) -> None:
    """A model built over a chosen slice of true entities keeps its scores in range."""
    # Create source data
    linked = linked_sources_factory()
    all_true_sources = list(linked.true_entities)

    # Get sources based on config
    left_testkit = linked.sources[source_params["left_name"]]
    right_testkit = (
        linked.sources[source_params["right_name"]]
        if source_params["right_name"]
        else None
    )

    # Create model through the testkit that planted the truth
    truth = all_true_sources[source_params["true_entities_slice"]]
    if source_params["right_name"]:
        model = linked.link(
            source_params["left_name"],
            source_params["right_name"],
            true_entities=truth,
            score_range=source_params["score_range"],
        )
    else:
        model = linked.dedupe(
            source_params["left_name"],
            true_entities=truth,
            score_range=source_params["score_range"],
        )

    assert str(model.model.model_type) == expected_checks["type"]
    assert (model.right_source is not None) == expected_checks["has_right"]

    score_values = model.scores["score"].to_numpy()
    assert all(p >= expected_checks["score_min"] for p in score_values)
    assert all(p <= expected_checks["score_max"] for p in score_values)

    # A partial slice of truth must still keep every source key in the model's output
    input_keys = sum(
        set(left_testkit.input_clusters)
        | set(right_testkit.input_clusters if right_testkit else {})
    )
    assert input_keys == sum(model.predicted_clusters), (
        "Model entities should preserve all source keys"
    )


@pytest.mark.parametrize(
    ("seed1", "seed2", "should_be_equal"),
    [
        pytest.param(42, 42, True, id="same_seeds"),
        pytest.param(1, 2, False, id="different_seeds"),
    ],
)
def test_generated_model_seed_behaviour(
    seed1: int, seed2: int, should_be_equal: bool
) -> None:
    """The same seed reproduces the data and scores. A different one changes both."""
    dummy1 = linked_sources_factory(seed=seed1).dedupe("crn", seed=seed1)
    dummy2 = linked_sources_factory(seed=seed2).dedupe("crn", seed=seed2)

    if should_be_equal:
        assert dummy1.left_source.data.equals(dummy2.left_source.data)
        assert set(dummy1.left_source.input_clusters) == set(
            dummy2.left_source.input_clusters
        )
        assert set(dummy1.predicted_clusters) == set(dummy2.predicted_clusters)
        assert dummy1.scores.equals(dummy2.scores)
    else:
        assert not dummy1.left_source.data.equals(dummy2.left_source.data)
        assert set(dummy1.left_source.input_clusters) != set(
            dummy2.left_source.input_clusters
        )
        assert set(dummy1.predicted_clusters) != set(dummy2.predicted_clusters)
        assert not dummy1.scores.equals(dummy2.scores)


def test_default_linker_sources_share_one_warehouse() -> None:
    """Both sides of a default linker must live in the same database.

    Regression test. `SourceParameters.engine` defaulted to an engine built once
    at class-definition time, and the `cdms` parameters omitted `engine=` while `crn`
    passed it, so the two sides of a link were written to different in-memory SQLite
    databases. A linker joins them, so they have to share one.
    """
    testkit = linked_sources_factory().link("crn", "cdms")

    sources = [step for step in testkit.model.lineage() if isinstance(step, Source)]
    assert {source.name for source in sources} == {"crn", "cdms"}

    locations = [source.location for source in sources]
    assert all(isinstance(loc, RelationalDB) for loc in locations)
    clients = {id(loc.client) for loc in locations}
    assert len(clients) == 1, "linked sources must share one warehouse"
