"""A linking methodology that applies different weights to field comparisons."""

from typing import ClassVar

import duckdb
import polars as pl
from pydantic import BaseModel, Field, field_validator

from matchlab.core.sql import SQLCondition
from matchlab.models import comparison
from matchlab.models.linkers.base import Linker


class WeightedComparison(BaseModel):
    """A comparison condition, and the weight it contributes to a pair's score."""

    comparison: SQLCondition = Field(
        description="""
            A valid ON clause comparing fields between the left and the right data.

            Qualify every column with `l` or `r`. For example:

            "l.company_name = r.company_name"
        """
    )
    weight: float = Field(
        description="""
            A weight to give this comparison. Use 1 for all comparisons to give
            uniform weight to each.
        """
    )

    @field_validator("comparison")
    @classmethod
    def validate_comparison(cls, v: SQLCondition) -> SQLCondition:
        """Validate the comparison string."""
        comp_val = comparison(v, dialect="duckdb")
        return comp_val


class WeightedDeterministicLinker(Linker):
    """Scores a pair by the weighted share of comparisons it matches on."""

    version: ClassVar[int] = 1

    weighted_comparisons: list[WeightedComparison] = Field(
        description="A list of tuples in the form of a comparison, and a weight."
    )
    threshold: float = Field(
        description="""
            The score above which matches will be kept. 
            
            Inclusive, so a value of 1 will keep only exact matches across all 
            comparisons.
        """,
        ge=0,
        le=1,
    )

    _id_dtype_l: pl.DataType
    _id_dtype_r: pl.DataType

    def prepare(self, left: pl.DataFrame, right: pl.DataFrame) -> None:
        """No preparation needed."""
        pass

    def link(self, left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
        """Score each pair by summed matching weight over total weight.

        Keeps only pairs scoring at or above `settings.threshold`.
        """
        self._id_dtype_l = left[self.left_id].dtype
        self._id_dtype_r = right[self.right_id].dtype

        # Used below but ruff can't detect
        left_df = left.clone()  # noqa: F841
        right_df = right.clone()  # noqa: F841

        match_subquery = []
        weights = []

        for weighted_comparison in self.weighted_comparisons:
            match_subquery.append(
                f"""
                    select distinct on (list_sort([raw.left_id, raw.right_id]))
                        raw.left_id,
                        raw.right_id,
                        1.0 * {weighted_comparison.weight} as score
                    from (
                        select
                            l.{self.left_id} as left_id,
                            r.{self.right_id} as right_id,
                        from
                            left_df l
                        inner join right_df r on
                            {weighted_comparison.comparison}
                    ) raw
                """
            )
            weights.append(weighted_comparison.weight)

        match_subquery = " union all ".join(match_subquery)
        total_weight = sum(weights)

        sql = f"""
            select
                matches.left_id,
                matches.right_id,
                sum(matches.score) / {total_weight} as score
            from
                ({match_subquery}) matches
            group by
                matches.left_id,
                matches.right_id
            having
                sum(matches.score) / 
                    {total_weight} >= {self.threshold};
        """

        return (
            duckdb.sql(sql)
            .pl()
            .with_columns(
                [
                    pl.col("left_id").cast(self._id_dtype_l),
                    pl.col("right_id").cast(self._id_dtype_r),
                    pl.col("score").cast(pl.Float32),
                ]
            )
        )
