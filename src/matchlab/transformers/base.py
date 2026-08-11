"""Base class for transformer methodologies, and the SQL they share.

A `Transformer` reshapes the records a model matches over. It is a pure function of a
frame: a `Transform` step calls `apply()` each time it collects. The frame always
carries an `id` column, and `apply()` must return a frame that still carries `id`, since
that is the grouping every downstream model and resolver reads.

Transformers are declarative and serialisable. Their fields *are* the configuration a
`Transform` folds into its spec and cache key, so keep them to plain data (no
callables), exactly as a `Deduper`'s settings are.
"""

from abc import ABC, abstractmethod

import duckdb
import polars as pl
from pydantic import BaseModel, ConfigDict
from sqlglot import expressions, parse_one
from sqlglot import select as sqlglot_select


class Transformer(BaseModel, ABC):
    """A reshaping of the records a model matches over.

    Concrete transformers (`Select`, `Clean`, `Group`) carry their configuration as
    flat fields, so `MyTransformer(...)` reads naturally where it is passed to
    `transform()`, and `model_dump(mode="json")` is the whole of its serialisation.
    """

    model_config = ConfigDict(frozen=True)

    @abstractmethod
    def apply(self, data: pl.DataFrame) -> pl.DataFrame:
        """Return `data` reshaped. Both the input and the output carry `id`."""
        ...


def _run(query: str, data: pl.DataFrame) -> pl.DataFrame:
    """Execute one DuckDB query against `data`, registered as `data`."""
    with duckdb.connect(":memory:") as connection:
        connection.register("data", data)
        return connection.execute(query).pl()


def apply_derive(data: pl.DataFrame, cleaning: dict[str, str]) -> pl.DataFrame:
    """Add or replace the named columns, keeping every other column (including `id`).

    Each value is a DuckDB SQL expression over the frame's columns, aliased to its key.
    An alias that names an existing column replaces it in place; a new alias is added.
    This is derivation, not projection: unreferenced columns survive untouched. Invalid
    SQL raises at build time, naming the expression.
    """
    collisions = [alias for alias in cleaning if alias in data.columns]
    star = expressions.Star(except_=[expressions.column(a) for a in collisions])
    projection: list[expressions.Expression] = [star]
    for alias, sql in cleaning.items():
        projection.append(expressions.alias_(parse_one(sql, dialect="duckdb"), alias))

    query = sqlglot_select(*projection, dialect="duckdb").from_("data")
    return _run(query.sql(dialect="duckdb"), data)


def apply_group(data: pl.DataFrame, aggregates: dict[str, str]) -> pl.DataFrame:
    """Collapse each `id` to one row, each named column an aggregate SQL expression.

    `id` is passed through as the grouping key. Every expression must be an aggregate;
    DuckDB reports a non-aggregate itself, naming the offending column. Only `id` and
    the named aggregates survive, since grouping has to say how every column combines.
    """
    identifier = expressions.alias_(expressions.column("id"), "id")
    projection: list[expressions.Expression] = [identifier]
    for alias, sql in aggregates.items():
        projection.append(expressions.alias_(parse_one(sql, dialect="duckdb"), alias))

    query = sqlglot_select(*projection, dialect="duckdb").from_("data").group_by("id")
    return _run(query.sql(dialect="duckdb"), data)
