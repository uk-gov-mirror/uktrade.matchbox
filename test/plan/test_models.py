"""Model-level helpers: the normalisation every methodology's scores go through."""

import polars as pl
from polars.testing import assert_frame_equal

from matchlab.models.models import normalise_model_scores


class TestModelProbabilities:
    """Test model score normalisation."""

    def test_duplicate_removal(self) -> None:
        """Removes redundant pairs, keeping highest score."""
        simple_duplicate = pl.DataFrame(
            [
                {"left_id": 4, "right_id": 5, "score": 0.5},
                {"left_id": 4, "right_id": 5, "score": 1.0},
            ]
        )

        assert_frame_equal(
            normalise_model_scores(simple_duplicate),
            simple_duplicate.tail(1),
            check_row_order=False,
            check_column_order=False,
            check_dtypes=False,
        )

        symmetric_duplicate = pl.DataFrame(
            [
                {"left_id": 5, "right_id": 4, "score": 0.5},
                {"left_id": 4, "right_id": 5, "score": 1.0},
            ]
        )

        assert_frame_equal(
            normalise_model_scores(symmetric_duplicate),
            symmetric_duplicate.tail(1),
            check_row_order=False,
            check_column_order=False,
            check_dtypes=False,
        )

        no_duplicates = pl.DataFrame(
            [
                {"left_id": 4, "right_id": 6, "score": 0.5},
                {"left_id": 4, "right_id": 5, "score": 1.0},
            ]
        )

        assert_frame_equal(
            normalise_model_scores(no_duplicates),
            no_duplicates,
            check_row_order=False,
            check_column_order=False,
            check_dtypes=False,
        )
