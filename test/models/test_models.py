"""Model-level helpers, including the normalisation every score goes through."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from matchlab.models.dedupers import NaiveDeduper
from matchlab.models.models import add_model_class, normalise_model_scores


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        pytest.param(
            pl.DataFrame(
                [
                    {"left_id": 4, "right_id": 5, "score": 0.5},
                    {"left_id": 4, "right_id": 5, "score": 1.0},
                ]
            ),
            pl.DataFrame([{"left_id": 4, "right_id": 5, "score": 1.0}]),
            id="a-pair-scored-twice-keeps-the-highest",
        ),
        pytest.param(
            pl.DataFrame(
                [
                    {"left_id": 5, "right_id": 4, "score": 0.5},
                    {"left_id": 4, "right_id": 5, "score": 1.0},
                ]
            ),
            pl.DataFrame([{"left_id": 4, "right_id": 5, "score": 1.0}]),
            id="a-reversed-pair-is-the-same-pair",
        ),
        pytest.param(
            pl.DataFrame(
                [
                    {"left_id": 4, "right_id": 6, "score": 0.5},
                    {"left_id": 4, "right_id": 5, "score": 1.0},
                ]
            ),
            pl.DataFrame(
                [
                    {"left_id": 4, "right_id": 6, "score": 0.5},
                    {"left_id": 4, "right_id": 5, "score": 1.0},
                ]
            ),
            id="distinct-pairs-are-both-kept",
        ),
    ],
)
def test_normalise_one_edge_per_pair(
    scores: pl.DataFrame, expected: pl.DataFrame
) -> None:
    """An unordered pair collapses to one edge, keeping the top score.

    `(a, b)` and `(b, a)` count as the same pair.
    """
    assert_frame_equal(
        normalise_model_scores(scores),
        expected,
        check_row_order=False,
        check_column_order=False,
        check_dtypes=False,
    )


def test_normalise_empty_input() -> None:
    """No edges is a valid answer, so an empty table normalises to the edge schema."""
    empty = pl.DataFrame(
        {"left_id": [], "right_id": [], "score": []},
        schema={"left_id": pl.UInt64, "right_id": pl.UInt64, "score": pl.Float64},
    )
    result = normalise_model_scores(empty)
    assert result.height == 0
    assert set(result.columns) == {"left_id", "right_id", "score"}


def test_normalise_rejects_non_dataframe() -> None:
    """A methodology must hand back a table, not a list of dicts or None."""
    with pytest.raises(ValueError, match="Expected a polars DataFrame"):
        normalise_model_scores([{"left_id": 1, "right_id": 2, "score": 0.5}])


def test_normalise_rejects_wrong_columns() -> None:
    """Missing the score column names what was expected against what was found."""
    bad = pl.DataFrame({"left_id": [1], "right_id": [2]})  # no score
    with pytest.raises(ValueError, match="Expected.*score"):
        normalise_model_scores(bad)


def test_normalise_rejects_non_numeric_score() -> None:
    """A score column that isn't numeric can't be a probability."""
    bad = pl.DataFrame({"left_id": [1], "right_id": [2], "score": ["high"]})
    with pytest.raises(ValueError, match="numeric values in the range"):
        normalise_model_scores(bad)


@pytest.mark.parametrize(
    "score",
    [
        pytest.param(1.5, id="above-one"),
        pytest.param(-0.5, id="below-zero"),
        pytest.param(float("nan"), id="nan"),
    ],
)
def test_normalise_rejects_out_of_range_score(score: float) -> None:
    """A score outside `[0.0, 1.0]` is a misconfigured methodology, not a match."""
    bad = pl.DataFrame({"left_id": [1], "right_id": [2], "score": [score]})
    with pytest.raises(ValueError, match="Score range misconfigured"):
        normalise_model_scores(bad)


def test_add_model_class_rejects_non_subclass() -> None:
    """Only a Deduper or Linker subclass can be registered as a model class."""
    with pytest.raises(ValueError, match="not a subclass of Deduper or Linker"):
        add_model_class(str)


def test_add_model_class_registers_a_subclass() -> None:
    """A real subclass registers under its name, so a plan can name it as a string."""
    add_model_class(NaiveDeduper)  # already known, so this call is idempotent
    from matchlab.models.models import _MODEL_CLASSES  # noqa: PLC0415

    assert _MODEL_CLASSES["NaiveDeduper"] is NaiveDeduper
