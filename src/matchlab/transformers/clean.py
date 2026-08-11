"""Clean derives columns. Add or replace named columns, keep the rest."""

import polars as pl
from pydantic import Field, field_validator

from matchlab.transformers.base import Transformer, apply_derive


class Clean(Transformer):
    """Derive columns with DuckDB SQL, without dropping unrelated fields.

    Each entry maps an output column name to a SQL expression over the frame's columns
    (the source-qualified names, `crn_company`). An alias that names an existing column
    replaces it. A new alias is added. Every other column, and `id`, passes through
    untouched. Dropping is `Select`'s job, not `Clean`'s.
    """

    cleaning: dict[str, str] = Field(
        description="Output column name to a DuckDB SQL expression over the frame."
    )

    @field_validator("cleaning")
    @classmethod
    def _non_empty(cls, cleaning: dict[str, str]) -> dict[str, str]:
        """A clean that derives nothing is meaningless. 'No cleaning' is no `Clean`."""
        if not cleaning:
            raise ValueError("Clean needs at least one expression to derive.")
        return cleaning

    def apply(self, data: pl.DataFrame) -> pl.DataFrame:
        """Add or replace the named columns, keeping the rest."""
        return apply_derive(data, self.cleaning)
