"""Group — granularity. Collapse each `id` to one row."""

import polars as pl
from pydantic import Field, field_validator

from matchlab.transformers.base import Transformer, apply_group


class Group(Transformer):
    """Collapse each `id` to a single row, combining columns with aggregate SQL.

    Reading several records per entity (through a resolver) gives a model more evidence,
    but a comparison needs one row. `Group` says how each column combines: `id` is the
    grouping key, and every named expression is a DuckDB aggregate,
    `any_value(crn_company)` where records agree, `list(distinct crn_town)` where they
    don't. Only `id` and the named aggregates survive, since grouping has to decide how
    every column collapses.
    """

    aggregates: dict[str, str] = Field(
        description="Output column name to a DuckDB aggregate expression."
    )

    @field_validator("aggregates")
    @classmethod
    def _non_empty(cls, aggregates: dict[str, str]) -> dict[str, str]:
        """Grouping collapses rows, so each column must say how it combines."""
        if not aggregates:
            raise ValueError(
                "Group needs aggregate expressions: grouping collapses several records "
                "into one row, so each column has to say how it combines. Pass "
                "aggregates such as {'name': 'any_value(crn_company)'}."
            )
        return aggregates

    def apply(self, data: pl.DataFrame) -> pl.DataFrame:
        """Collapse each `id` to one row using the aggregate expressions."""
        return apply_group(data, self.aggregates)
