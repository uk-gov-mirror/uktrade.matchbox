"""Model-level helpers: the normalisation every methodology's scores go through."""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from matchlab.models.models import normalise_model_scores


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
    """One edge per unordered pair: `(a, b)` and `(b, a)` collapse, top score wins."""
    assert_frame_equal(
        normalise_model_scores(scores),
        expected,
        check_row_order=False,
        check_column_order=False,
        check_dtypes=False,
    )
