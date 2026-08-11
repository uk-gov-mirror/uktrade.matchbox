"""Interface to locations where source data is stored.

Location classes are registered by name.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from enum import StrEnum
from typing import Any, ClassVar, Literal, overload

import polars as pl
import sqlglot
from adbc_driver_manager.dbapi import Connection as AdbcConnection
from pandas import DataFrame as PandasDataFrame
from polars import DataFrame as PolarsDataFrame
from pyarrow import Table as ArrowTable
from sqlalchemy import Connection, Engine
from sqlalchemy.exc import OperationalError
from sqlglot.errors import ParseError

from matchlab.core.dataframes import (
    DataFrameClass,
    DataFrameType,
    sql_to_df,
)
from matchlab.core.exceptions import ExtractTransformError
from matchlab.core.logging import logger
from matchlab.core.sql import SQLQuery


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


def build_location(location_class: str, name: str, client: Any) -> "Location":  # noqa: ANN401
    """Build a registered location class, bound to a name and a client.

    How `matchlab.document` rebuilds the locations a plan reads. Both arguments come
    from outside any spec: the class because a `SourceSpec` deliberately says
    nothing about where its rows came from, the client because a document carries no
    credentials.

    Raises:
        ValueError: If no location class of that name is registered here.
    """
    if location_class not in _LOCATION_CLASSES:
        raise ValueError(
            f"No location class named '{location_class}' is registered. Register it "
            "with `add_location_class` before loading a document that names it."
        )
    return _LOCATION_CLASSES[location_class](name=name, client=client)


class Location(ABC):
    """A named place data is read from, bound to the client that reads it."""

    client_classes: ClassVar[tuple[type, ...]]
    """The client types this location can be driven by, checked on construction."""

    def __init__(self, name: str, client: Any) -> None:  # noqa: ANN401
        """Initialise a location.

        Args:
            name: How a plan refers to this location. A handle for the client, not a
                setting. It never reaches a fingerprint.
            client: The already-configured client to read through. Required: a location
                without one can answer nothing, not even which SQL dialect it speaks.

        Raises:
            ValueError: If the client is not a type this location can use.
        """
        if not isinstance(client, self.client_classes):
            raise ValueError(
                f"{type(client).__name__} is not a valid client for "
                f"{type(self).__name__}. Expected one of: "
                f"{', '.join(accepted.__name__ for accepted in self.client_classes)}."
            )
        self.name = name
        self._client = client

    def __str__(self) -> str:
        """Identify the location by class and name."""
        return f"{type(self).__name__} '{self.name}'"

    @property
    def client(self) -> Any:  # noqa: ANN401
        """The client this location reads through. Set once, at construction."""
        return self._client

    @abstractmethod
    def connect(self) -> bool:
        """Check the location can actually be reached."""
        ...

    @abstractmethod
    def validate_extract_transform(self, extract_transform: SQLQuery) -> None:
        """Validate ET logic against this location's query language.

        Raises:
            ExtractTransformError: If the ET logic is invalid.
        """
        ...

    @abstractmethod
    def execute(
        self,
        extract_transform: SQLQuery,
        batch_size: int | None = None,
        rename: dict[str, str] | Callable | None = None,
        return_type: DataFrameType = DataFrameType.POLARS,
        keys: tuple[str, list[str]] | None = None,
        schema_overrides: dict[str, pl.DataType] | None = None,
    ) -> Iterator[DataFrameClass]:
        """Execute ET logic against location and return batches.

        Args:
            extract_transform: The ET logic to execute.
            batch_size: The size used for internal batching. Overrides environment
                variable if set.
            rename: Renaming to apply after the ET logic is executed.

                * If a dictionary is provided, it will be used to rename the columns.
                * If a callable is provided, it will take the old name as input and
                    return the new name.
            return_type: The type of data to return. Defaults to "polars".
            keys: Rule to only retrieve rows by specific keys.
                The key of the dictionary is a field name on which to filter.
                Filters source entries where the key field is in the dict values.
            schema_overrides: Types to force on the columns that come back, rather
                than letting the client infer them.
        """
        ...


class RelationalDBLocation(Location):
    """A location for a relational database."""

    client_classes: ClassVar[tuple[type, ...]] = (Engine, AdbcConnection)

    @property
    def client(self) -> Engine | AdbcConnection:
        """The SQLAlchemy engine or ADBC connection this location reads through."""
        return self._client

    @property
    def client_type(self) -> ClientType:
        """Which client type this location was built with.

        One `isinstance` decides it, because `client_classes` has already rejected
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

    def connect(self) -> bool:  # noqa: D102
        try:
            with self._get_connection() as conn:
                _ = sql_to_df(stmt="select 1", connection=conn)
                return True
        except OperationalError:
            return False

    @property
    def _sqlglot_dialect(self) -> str | None:
        """The dialect to parse extract/transform SQL as, if we recognise it."""
        if self.client_type == ClientType.SQLALCHEMY:
            client_dialect = self.client.dialect.name
        else:
            vendor = self.client.adbc_get_info().get("vendor_name")
            client_dialect = vendor.lower() if vendor else None

        match client_dialect:
            case "postgresql":
                return "postgres"
            case "sqlite":
                return "sqlite"
            case _:
                logger.warning("Could not validate specific dialect.")
                return None

    def validate_extract_transform(self, extract_transform: SQLQuery) -> None:
        """Check that the SQL statement only contains a single data-extracting command.

        This does not fully sanitise the SQL statement. Validation only guards
        against accidental mistakes, not malicious actors. Only run indexing using
        sources you trust and have read, with least-privilege credentials.

        Args:
            extract_transform: The SQL statement to validate.

        Raises:
            ParseError: If the SQL statement cannot be parsed.
            ExtractTransformError: If validation requirements are not met.
        """
        if not extract_transform.strip():
            raise ExtractTransformError(
                "SQL statement is empty or only contains whitespace."
            )

        try:
            expressions = sqlglot.parse(
                extract_transform, dialect=self._sqlglot_dialect
            )
        except ParseError as e:
            raise ExtractTransformError("SQL statement could not be parsed.") from e

        if len(expressions) > 1:
            raise ExtractTransformError("SQL statement contains multiple commands.")

        if not expressions:
            raise ExtractTransformError(
                "SQL statement does not contain any valid expressions."
            )

        expr = expressions[0]

        if not isinstance(expr, sqlglot.expressions.Select | sqlglot.expressions.Union):
            raise ExtractTransformError(
                "SQL statement must start with a SELECT or WITH command."
            )

        forbidden = (
            sqlglot.expressions.DDL,
            sqlglot.expressions.DML,
            sqlglot.expressions.Into,
        )

        if len(list(expr.find_all(forbidden))) > 0:
            raise ExtractTransformError(
                "SQL statement must not contain DDL or DML commands."
            )

    @overload
    def execute(
        self,
        extract_transform: SQLQuery,
        batch_size: int | None = None,
        rename: dict[str, str] | Callable | None = None,
        return_type: Literal[DataFrameType.POLARS] = ...,
        keys: tuple[str, list[str]] | None = None,
        schema_overrides: dict[str, pl.DataType] | None = None,
    ) -> Generator[PolarsDataFrame, None, None]: ...

    @overload
    def execute(
        self,
        extract_transform: SQLQuery,
        batch_size: int | None = None,
        rename: dict[str, str] | Callable | None = None,
        return_type: Literal[DataFrameType.PANDAS] = ...,
        keys: tuple[str, list[str]] | None = None,
        schema_overrides: dict[str, pl.DataType] | None = None,
    ) -> Generator[PandasDataFrame, None, None]: ...

    @overload
    def execute(
        self,
        extract_transform: SQLQuery,
        batch_size: int | None = None,
        rename: dict[str, str] | Callable | None = None,
        return_type: Literal[DataFrameType.ARROW] = ...,
        keys: tuple[str, list[str]] | None = None,
        schema_overrides: dict[str, pl.DataType] | None = None,
    ) -> Generator[ArrowTable, None, None]: ...

    def execute(  # noqa: D102
        self,
        extract_transform: SQLQuery,
        batch_size: int | None = None,
        rename: dict[str, str] | Callable | None = None,
        return_type: DataFrameType = DataFrameType.POLARS,
        keys: tuple[str, list[str]] | None = None,
        schema_overrides: dict[str, pl.DataType] | None = None,
    ) -> Generator[DataFrameClass, None, None]:
        # Strip semicolon, as it can block extended query protocol
        # and slow some engines down
        extract_transform = extract_transform.replace(";", "")

        # We always work in batches to simplify higher level logic
        batch_size: int = batch_size or 10_000

        # TODO(adbc-batching): this size is honoured for SQLAlchemy clients and
        # silently ignored for ADBC ones. Polars registers ADBC with
        # `exact_batch_size: False` (see `polars/io/database/_arrow_registry.py`), so
        # it calls `fetch_record_batch()` with no size and the driver chunks however
        # it likes. In-process drivers return the whole result in one batch.
        # `Source.sample(n)` takes the first batch and so can return the entire
        # source, and `_read_warehouse`'s stream-to-Parquet stops bounding memory.
        # Fix by slicing the record-batch stream to `batch_size` in `sql_to_df`.

        with self._get_connection() as conn:
            if keys:
                key_field, filter_values = keys
                quoted_key_values = [f"'{key_val}'" for key_val in filter_values]
                # Only filter original SQL if keys are provided
                if quoted_key_values:
                    comma_separated_values = ", ".join(quoted_key_values)
                    # This "IN" expression is a SQL standard, though that is hard to
                    # prove, since the standard is behind a paywall.
                    extract_transform = (
                        f"select * from ({extract_transform}) as sub "
                        f"where {key_field} in ({comma_separated_values})"
                    )
            yield from sql_to_df(
                stmt=extract_transform,
                schema_overrides=schema_overrides,
                connection=conn,
                rename=rename,
                batch_size=batch_size,
                return_batches=True,
                return_type=return_type,
            )


add_location_class(RelationalDBLocation)
