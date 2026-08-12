"""Base class for transformer methodologies, and the DuckDB query runner some share.

A `Transformer` is a pure function of a record step, called by a `Transform` step's
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

from matchlab.core.sql import SQLQuery


class Transformer(BaseModel, ABC):
    """Base contract every transformer implements.

    Concrete transformers (`Select`, `Clean`, `Group`, `Explode`) carry their
    configuration as flat fields, so `MyTransformer(...)` reads naturally, and
    `model_dump(mode="json")` is the whole of its serialisation.
    """

    model_config = ConfigDict(frozen=True)

    @abstractmethod
    def apply(self, data: pl.DataFrame) -> pl.DataFrame:
        """Return `data` reshaped. Both the input and the output carry `id`."""
        ...


def run_sql(query: SQLQuery, data: pl.DataFrame) -> pl.DataFrame:
    """Execute one DuckDB query against `data`, registered as `data`."""
    with duckdb.connect(":memory:") as connection:
        connection.register("data", data)
        return connection.execute(query).pl()
