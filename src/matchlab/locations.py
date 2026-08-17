"""Interface to the places source data is read from.

A `Location` is a methodology, exactly as a `Deduper` or a `Transformer` is. A `Source`
names the class and hands it settings and resources separately, so the query travels in
a document and the client never does. See `matchlab.resources`.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from enum import StrEnum
from typing import TypeAlias, cast

import polars as pl
from adbc_driver_manager.dbapi import Connection as AdbcConnection
from pandas import DataFrame as PandasDataFrame
from polars import DataFrame as PolarsDataFrame
from pydantic import BaseModel, ConfigDict
from sqlalchemy import Connection, Engine

from matchlab.core.dataframes import (
    DataFrameClass,
    DataFrameType,
    sql_to_df,
    to_dataframe,
)
from matchlab.core.sql import SQLQuery
from matchlab.resources import FromResources

DBClient: TypeAlias = Engine | AdbcConnection
"""What a `RelationalDB` reads through: a SQLAlchemy engine or an ADBC connection."""


class ClientType(StrEnum):
    """The client libraries a relational location can be driven by."""

    SQLALCHEMY = "sqlalchemy"
    ADBC = "adbc"


_LOCATION_CLASSES: dict[str, "type[Location]"] = {}


def add_location_class(location_class: "type[Location]") -> None:
    """Register a custom location so it can be named in a plan document."""
    if not issubclass(location_class, Location):
        raise ValueError("The argument is not a subclass of Location.")
    _LOCATION_CLASSES[location_class.__name__] = location_class


def resolve_location_class(location_class: "str | type[Location]") -> "type[Location]":
    """Return a location class, looking a name up in the registry.

    How `Source` accepts either the class or its registered name, and how
    `matchlab.document` rebuilds the locations a plan reads.

    Raises:
        ValueError: If no location class of that name is registered here.
    """
    if not isinstance(location_class, str):
        return location_class
    if location_class not in _LOCATION_CLASSES:
        raise ValueError(
            f"No location class named '{location_class}' is registered. Register it "
            "with `add_location_class` before loading a document that names it."
        )
    return _LOCATION_CLASSES[location_class]


class Location(BaseModel, ABC):
    """A place data is read from, holding everything needed to read it.

    Every field is a setting unless marked `matchlab.resources.FromResources`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    def __str__(self) -> str:
        """Identify the location by class."""
        return type(self).__name__

    @abstractmethod
    def read(
        self,
        batch_size: int | None = None,
        rename: dict[str, str] | Callable | None = None,
        return_type: DataFrameType = DataFrameType.POLARS,
        schema_overrides: dict[str, pl.DataType] | None = None,
    ) -> Iterator[DataFrameClass]:
        """Get this location's rows, in batches.

        Args:
            batch_size: The size used for internal batching.
            rename: Renaming to apply to the rows that come back.

                * If a dictionary is provided, it will be used to rename the columns.
                * If a callable is provided, it will take the old name as input and
                    return the new name.
            return_type: The type of data to return. Defaults to "polars".
            schema_overrides: Types to force on the columns that come back, rather than
                letting the location infer them.
        """
        ...


class RelationalDB(Location):
    """A relational database, and the query that extracts a source's rows from it."""

    sql: SQLQuery
    """SQL producing the rows to read, in whatever dialect the client speaks."""

    client: FromResources[DBClient]
    """The SQLAlchemy engine or ADBC connection to read through."""

    @property
    def client_type(self) -> ClientType:
        """Which client type this location was built with.

        One `isinstance` decides it, because the field's type has already rejected
        anything that is neither an `Engine` nor an ADBC connection.
        """
        return (
            ClientType.SQLALCHEMY
            if isinstance(self.client, Engine)
            else ClientType.ADBC
        )

    @contextmanager
    def _get_connection(self) -> Generator[Connection | AdbcConnection, None, None]:
        """Get a database connection, and close or release it properly afterwards."""
        if self.client_type == ClientType.ADBC:
            # Cloning avoids memory corruption in the ADBC driver, which happens if
            # a single connection is reused for multiple queries.
            with self.client.adbc_clone() as connection:
                yield connection
        else:  # SQLAlchemy Engine
            connection = self.client.connect()
            try:
                yield connection
            finally:
                connection.close()

    def read(  # noqa: D102
        self,
        batch_size: int | None = None,
        rename: dict[str, str] | Callable | None = None,
        return_type: DataFrameType = DataFrameType.POLARS,
        schema_overrides: dict[str, pl.DataType] | None = None,
    ) -> Generator[DataFrameClass, None, None]:
        # Strip semicolon, as it can block extended query protocol
        # and slow some engines down
        statement = self.sql.replace(";", "")

        # We always work in batches to simplify higher level logic
        batch_size: int = batch_size or 10_000

        # TODO(adbc-batching): this size is honoured for SQLAlchemy clients and
        # silently ignored for ADBC ones. Polars registers ADBC with
        # `exact_batch_size: False` (see `polars/io/database/_arrow_registry.py`), so
        # it calls `fetch_record_batch()` with no size and the driver chunks however
        # it likes. In-process drivers return the whole result in one batch.
        # `Source.sample(n)` takes the first batch and so can return the entire
        # source, and `_read_origin`'s stream-to-Parquet stops bounding memory.
        # Fix by slicing the record-batch stream to `batch_size` in `sql_to_df`.

        with self._get_connection() as conn:
            yield from sql_to_df(
                stmt=statement,
                schema_overrides=schema_overrides,
                connection=conn,
                rename=rename,
                batch_size=batch_size,
                return_batches=True,
                return_type=return_type,
            )


class DataFrame(Location):
    """A dataframe already in memory."""

    df: FromResources[DataFrameClass]
    """The rows to read."""

    @property
    def _polars(self) -> PolarsDataFrame:
        """The frame as polars, whatever it was handed as."""
        if isinstance(self.df, PolarsDataFrame):
            return self.df
        if isinstance(self.df, PandasDataFrame):
            return pl.from_pandas(self.df)
        # An arrow `Table` always converts to a frame; only a `ChunkedArray` would
        # give a `Series`, and the field type excludes one.
        return cast(PolarsDataFrame, pl.from_arrow(self.df))

    def read(  # noqa: D102
        self,
        batch_size: int | None = None,
        rename: dict[str, str] | Callable | None = None,
        return_type: DataFrameType = DataFrameType.POLARS,
        schema_overrides: dict[str, pl.DataType] | None = None,
    ) -> Generator[DataFrameClass, None, None]:
        frame = self._polars

        if schema_overrides:
            frame = frame.with_columns(
                pl.col(column).cast(dtype)
                for column, dtype in schema_overrides.items()
                if column in frame.columns
            )

        if rename:
            frame = frame.rename(rename)

        batch_size = batch_size or 10_000
        for offset in range(0, frame.height, batch_size):
            yield to_dataframe(frame.slice(offset, batch_size), return_type)


add_location_class(RelationalDB)
add_location_class(DataFrame)
