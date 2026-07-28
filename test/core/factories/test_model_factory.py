from typing import Any, Literal

import polars as pl
import pytest

from matchlab.core.arrow import SCHEMA_MODEL_EDGES
from matchlab.core.factories.models import (
    generate_dummy_scores,
    model_factory,
)
from matchlab.core.factories.sources import linked_sources_factory, source_factory


@pytest.mark.parametrize(
    ("left_testkit", "right_testkit", "expected_type", "should_have_right"),
    [
        pytest.param(None, None, "deduper", False, id="default_creates_deduper"),
        pytest.param(
            "source", None, "deduper", False, id="left_source_only_creates_deduper"
        ),
        pytest.param(
            "source", "source", "linker", True, id="both_sources_creates_linker"
        ),
    ],
)
def test_model_type_creation(
    left_testkit: None | str,
    right_testkit: None | str,
    expected_type: str,
    should_have_right: bool,
) -> None:
    """Test that model creation and core operations work correctly for each type."""
    # Create our source objects from the string parameters
    linked = linked_sources_factory()
    all_true_sources = list(linked.true_entities)

    left = linked.sources["crn"] if left_testkit == "source" else None
    right = linked.sources["cdms"] if right_testkit == "source" else None

    # Create our model
    model = model_factory(
        left_testkit=left, right_testkit=right, true_entities=all_true_sources, seed=13
    )

    # Basic type verification
    assert model.model.model_type == expected_type
    assert (model.right is not None) == should_have_right
    assert (model.right_clusters is not None) == should_have_right

    # Verify scores were generated
    assert len(model.scores) > 0
    assert model.scores.schema == pl.Schema(SCHEMA_MODEL_EDGES)

    # Verify the input data exists and includes ids
    assert "id" in model.left_data.column_names
    assert len(set(model.left_data["id"].to_pylist())) > 0

    # For linkers, verify we maintain separation between left and right IDs
    if expected_type == "linker":
        left_ids = set(model.left_data["id"].to_pylist())
        right_ids = set(model.right_data["id"].to_pylist())
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
    """Tests the factories validate "real" methodologies in various pipeline positions.

    Here we show that with just a single output of a scores table, the factory
    and testkit system lets you evaluate the methodology of a deduper or linker.

    This test demonstrates that:
    1. We can set up pipelines in various configurations that work perfectly
        with model_factory
    2. When we swap in a simulated "real" methodology (using
        generate_dummy_scores), the diff can detect the errors appropriately
    3. This validation works across different pipeline positions and configurations
    """
    linked = linked_sources_factory()
    all_true_sources = list(linked.true_entities)

    # Create and validate perfect model
    if model_type == "deduper":
        # Get inputs to final model for later diff
        left_clusters = linked.sources[left_testkit].entities
        right_clusters = None

        perfect_model = model_factory(
            left_testkit=linked.sources[left_testkit],
            true_entities=all_true_sources,
        )
        sources = [left_testkit]
        model_entities = (tuple(linked.sources[left_testkit].entities), None)
    else:  # linker
        # Get inputs to final model for later diff
        left_clusters = linked.sources[left_testkit].entities
        right_clusters = linked.sources[right_testkit].entities

        perfect_model = model_factory(
            left_testkit=linked.sources[left_testkit],
            right_testkit=linked.sources[right_testkit],
            true_entities=all_true_sources,
        )
        sources = [left_testkit, right_testkit]
        model_entities = (
            tuple(linked.sources[left_testkit].entities),
            tuple(linked.sources[right_testkit].entities),
        )

    # Verify perfect model works
    identical, _ = linked.diff_model_edges(
        scores=perfect_model.scores,
        left_clusters=left_clusters,
        right_clusters=right_clusters,
        sources=sources,
        threshold=0,
    )
    assert identical, "Perfect model_factory setup should match"

    # Test with imperfect methodology
    random_scores = generate_dummy_scores(
        left_values=tuple(c.id for c in model_entities[0]),
        right_values=tuple(c.id for c in model_entities[1])
        if model_entities[1] is not None
        else None,
        score_range=(0.0, 1.0),
        num_components=len(all_true_sources) - 1,  # Intentionally wrong
    )

    identical, report = linked.diff_model_edges(
        scores=random_scores,
        left_clusters=left_clusters,
        right_clusters=right_clusters,
        sources=sources,
        threshold=0,
    )

    # Verify the imperfect methodology was detected
    assert not identical
    # Random process: can't guarantee particular issues, but can guarantee
    # that some will be present
    assert report["wrong"] > 0 or report["subset"] > 0 or report["superset"] > 0


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
        pytest.param(
            {"left_testkit": source_factory(), "true_entities": None},
            ValueError,
            "Must provide true entities when sources are given",
            id="missing_true_entities_with_source",
        ),
    ],
)
def test_model_factory_validation(
    kwargs: dict[str, Any], expected_error: type[Exception], expected_message: str
) -> None:
    """Test that model_factory validates inputs correctly."""
    with pytest.raises(expected_error, match=expected_message):
        model_factory(**kwargs)


@pytest.mark.parametrize(
    (
        "name",
        "model_type",
        "n_true_entities",
        "score_range",
        "seed",
        "expected_checks",
    ),
    [
        pytest.param(
            "basic_deduper",
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
            "basic_linker",
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
            "large_deduper",
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
            "strict_linker",
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
def test_model_factory_basic_creation(
    name: str,
    model_type: str,
    n_true_entities: int,
    score_range: tuple[float, float],
    seed: int,
    expected_checks: dict,
) -> None:
    """Test basic model factory creation without sources."""
    model = model_factory(
        name=name,
        model_type=model_type,
        n_true_entities=n_true_entities,
        score_range=score_range,
        seed=seed,
    )

    # Basic metadata checks
    assert model.name == name
    assert str(model.model.model_type) == expected_checks["type"]

    # Structure checks
    assert (model.right is not None) == expected_checks["has_right"]
    assert (model.right_clusters is not None) == expected_checks["has_right"]
    assert len(model.entities) == expected_checks["entity_count"]

    # Probability checks
    score_values = model.scores["score"].to_numpy()
    assert all(p >= expected_checks["score_min"] for p in score_values)
    assert all(p <= expected_checks["score_max"] for p in score_values)


@pytest.mark.parametrize(
    ("source_config", "expected_checks"),
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
def test_model_factory_with_sources(source_config: dict, expected_checks: dict) -> None:
    """Test model factory creation using sources."""
    # Create source data
    linked = linked_sources_factory()
    all_true_sources = list(linked.true_entities)

    # Get sources based on config
    left_testkit = linked.sources[source_config["left_name"]]
    right_testkit = (
        linked.sources[source_config["right_name"]]
        if source_config["right_name"]
        else None
    )

    # Create model
    model = model_factory(
        left_testkit=left_testkit,
        right_testkit=right_testkit,
        true_entities=all_true_sources[source_config["true_entities_slice"]],
        score_range=source_config["score_range"],
    )

    # Basic type checks
    assert str(model.model.model_type) == expected_checks["type"]
    assert (model.right is not None) == expected_checks["has_right"]
    assert (model.right_clusters is not None) == expected_checks["has_right"]

    # Verify scores
    score_values = model.scores["score"].to_numpy()
    assert all(p >= expected_checks["score_min"] for p in score_values)
    assert all(p <= expected_checks["score_max"] for p in score_values)

    # Verify source keys are preserved
    input_keys = sum(
        set(left_testkit.entities)
        | set(right_testkit.entities if right_testkit else {})
    )
    assert input_keys == sum(model.entities), (
        "Model entities should preserve all source keys"
    )


@pytest.mark.parametrize(
    ("seed1", "seed2", "should_be_equal"),
    [
        pytest.param(42, 42, True, id="same_seeds"),
        pytest.param(1, 2, False, id="different_seeds"),
    ],
)
def test_model_factory_seed_behavior(
    seed1: int, seed2: int, should_be_equal: bool
) -> None:
    """Test that model_factory handles seeds correctly for reproducibility."""
    dummy1 = model_factory(seed=seed1)
    dummy2 = model_factory(seed=seed2)

    if should_be_equal:
        assert dummy1.name == dummy2.name
        assert dummy1.left_data.equals(dummy2.left_data)
        assert set(dummy1.left_clusters) == set(dummy2.left_clusters)
        assert set(dummy1.entities) == set(dummy2.entities)
        assert dummy1.scores.equals(dummy2.scores)
    else:
        assert dummy1.name != dummy2.name
        assert not dummy1.left_data.equals(dummy2.left_data)
        assert set(dummy1.left_clusters) != set(dummy2.left_clusters)
        assert set(dummy1.entities) != set(dummy2.entities)
        assert not dummy1.scores.equals(dummy2.scores)
