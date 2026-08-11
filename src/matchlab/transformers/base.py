"""Base class for transformer methodologies, and the SQL they share.

A `Transformer` is a pure function of a frame, called by a `Transform` step's
`apply()` on every collect. Both input and output must carry `id`, the grouping every
downstream model and resolver reads.

Transformers are declarative and serialisable. Their fields are the configuration a
`Transform` folds into its spec and cache key, so keep them to plain data, with no
callables, exactly as a `Deduper`'s settings are.
"""

from abc import ABC, abstractmethod

import duckdb
import polars as pl
from pydantic import BaseModel, ConfigDict
from sqlglot import expressions, parse_one
from sqlglot import select as sqlglot_select


class Transformer(BaseModel, ABC):
    """Base contract every transformer implements.

    Concrete transformers (`Select`, `Clean`, `Group`) carry their configuration as
    flat fields, so `MyTransformer(...)` reads naturally, and `model_dump(mode="json")`
    is the whole of its serialisation.
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
    An alias that names an existing column replaces it in place. A new alias is added.
    This is derivation, not projection, so unreferenced columns survive untouched.
    Invalid SQL raises at build time, naming the expression.
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

    `id` is passed through as the grouping key. Every expression must be an aggregate.
    DuckDB itself reports a non-aggregate, naming the offending column. Only `id` and
    the named aggregates survive, since grouping has to say how every column combines.
    """
    identifier = expressions.alias_(expressions.column("id"), "id")
    projection: list[expressions.Expression] = [identifier]
    for alias, sql in aggregates.items():
        projection.append(expressions.alias_(parse_one(sql, dialect="duckdb"), alias))

    query = sqlglot_select(*projection, dialect="duckdb").from_("data").group_by("id")
    return _run(query.sql(dialect="duckdb"), data)
