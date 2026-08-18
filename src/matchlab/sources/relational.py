"""Reading a source from a relational database."""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from enum import StrEnum
from typing import TypeAlias

import polars as pl
from adbc_driver_manager.dbapi import Connection as AdbcConnection
from sqlalchemy import Connection, Engine

from matchlab.core.dataframes import DataFrameClass, DataFrameType, sql_to_df
from matchlab.core.sql import SQLQuery
from matchlab.resources import FromResources
from matchlab.sources.base import Location

DBClient: TypeAlias = Engine | AdbcConnection
"""What a `RelationalDB` reads through: a SQLAlchemy engine or an ADBC connection."""


class ClientType(StrEnum):
    """The client libraries a relational location can be driven by."""

    SQLALCHEMY = "sqlalchemy"
    ADBC = "adbc"


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
