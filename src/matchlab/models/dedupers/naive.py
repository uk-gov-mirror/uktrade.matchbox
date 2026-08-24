"""A deduplication methodology based on a deterministic set of conditions."""

from typing import ClassVar

import duckdb
import polars as pl
from pydantic import Field

from matchlab.models.dedupers.base import Deduper


class NaiveDeduper(Deduper):
    """Groups records that match exactly on every field in `unique_fields`."""

    version: ClassVar[int] = 1

    unique_fields: list[str] = Field(
        description="A list of fields that will form a unique, deduplicated record"
    )

    _id_dtype: pl.DataType

    def prepare(self, data: pl.DataFrame) -> None:
        """No preparation needed."""
        pass

    def dedupe(self, data: pl.DataFrame) -> pl.DataFrame:
        """Deduplicate the dataframe. Every match scores 1.0."""
        self._id_dtype = data[self.id].dtype

        df = data.clone()

        join_clause = []
        for field in self.unique_fields:
            join_clause.append(f"l.{field} = r.{field}")
        join_clause_compiled = " and ".join(join_clause)

        # Add a row index so a genuine duplicate row (identical values, different
        # index) isn't discarded as a self-match.
        df = df.with_row_index("_unique_e4003b")

        # If `id` isn't unique per row (for example, after unnesting an array of
        # keys), two different rows can still share the same left_id/right_id value.
        # Drop those pairs too.
        sql = f"""
            select distinct on (list_sort([raw.left_id, raw.right_id]))
                raw.left_id,
                raw.right_id,
                1.0 as score
            from (
                select
                    l.{self.id} as left_id,
                    r.{self.id} as right_id
                from
                    df l
                inner join df r on
                    (
                        {join_clause_compiled}
                    ) and
                        l._unique_e4003b != r._unique_e4003b
            ) raw
                where raw.left_id != raw.right_id;
        """

        return (
            duckdb.sql(sql)
            .pl()
            .with_columns(
                [
                    pl.col("left_id").cast(self._id_dtype),
                    pl.col("right_id").cast(self._id_dtype),
                    pl.col("score").cast(pl.Float32),
                ]
            )
        )
