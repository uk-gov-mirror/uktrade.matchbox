"""Dataframe types, conversion, column naming and SQL reading for matchlab."""

import re
from collections.abc import Callable, Iterator
from enum import StrEnum
from typing import (
    Any,
    Literal,
    TypeAlias,
    overload,
)

import polars as pl
from adbc_driver_manager.dbapi import Connection as AdbcConnection
from pandas import DataFrame as PandasDataFrame
from polars import DataFrame as PolarsDataFrame
from pyarrow import Table as ArrowTable
from sqlalchemy.engine import Engine

from matchlab.core.sql import SQLQuery


class DataFrameType(StrEnum):
    """Enumeration of dataframe types to return from query."""

    PANDAS = "pandas"
    POLARS = "polars"
    ARROW = "arrow"


DataFrameClass: TypeAlias = ArrowTable | PandasDataFrame | PolarsDataFrame

# A source's name prefixes every column it contributes (`crn` gives `crn_company`),
# and those land in cleaning SQL that sqlglot parses. `crn-x_company` would parse as
# subtraction and `crn.x_company` as table.column, so the name has to be usable at the
# start of an unquoted identifier. Reserved words are fine — the name is only ever a
# prefix, never a bare identifier.
_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def validate_col_prefix(prefix: str) -> None:
    """Ensure string is safe to use as a prefix in a dataframe column."""
    if not _IDENTIFIER.match(prefix):
        raise ValueError(
            f"'{prefix}' can't prefix a column name. It must start with a letter or"
            "underscore and contain only letters, digits and underscores."
        )


def qualify(source: str, field: str = "") -> str:
    """The column name a source's field carries once qualified.

    Args:
        source: The source's name.
        field: The field to qualify. Omit it for the bare prefix, which is what
            `pl.all().name.prefix()` takes.

    Returns:
        The qualified column name.
    """
    return f"{source}_{field}"


@overload
def to_dataframe(
    data: PolarsDataFrame, return_type: Literal[DataFrameType.POLARS]
) -> PolarsDataFrame: ...


@overload
def to_dataframe(
    data: PolarsDataFrame, return_type: Literal[DataFrameType.PANDAS]
) -> PandasDataFrame: ...


@overload
def to_dataframe(
    data: PolarsDataFrame, return_type: Literal[DataFrameType.ARROW]
) -> ArrowTable: ...


def to_dataframe(
    data: PolarsDataFrame, return_type: DataFrameType = DataFrameType.POLARS
) -> DataFrameClass:
    """Convert a Polars dataframe to the requested dataframe type.

    Args:
        data: The frame to convert.
        return_type: The dataframe type to return. One of "arrow", "pandas",
            or "polars".

    Returns:
        The data in the requested format.

    Raises:
        ValueError: If `return_type` isn't a member of `DataFrameType`.
    """
    match return_type:
        case DataFrameType.POLARS:
            return data
        case DataFrameType.PANDAS:
            return data.to_pandas()
        case DataFrameType.ARROW:
            return data.to_arrow()
        case _:
            raise ValueError(f"Return type {return_type} is invalid")


@overload
def sql_to_df(
    stmt: SQLQuery,
    connection: Engine | AdbcConnection,
    return_type: Literal[DataFrameType.POLARS],
    *,
    return_batches: Literal[False] = False,
    batch_size: int | None = None,
    rename: dict[str, str] | Callable | None = None,
    schema_overrides: dict[str, pl.DataType] | None = None,
    execute_options: dict[str, Any] | None = None,
) -> PolarsDataFrame: ...


@overload
def sql_to_df(
    stmt: SQLQuery,
    connection: Engine | AdbcConnection,
    return_type: Literal[DataFrameType.PANDAS],
    *,
    return_batches: Literal[False] = False,
    batch_size: int | None = None,
    rename: dict[str, str] | Callable | None = None,
    schema_overrides: dict[str, pl.DataType] | None = None,
    execute_options: dict[str, Any] | None = None,
) -> PandasDataFrame: ...


@overload
def sql_to_df(
    stmt: SQLQuery,
    connection: Engine | AdbcConnection,
    return_type: Literal[DataFrameType.ARROW],
    *,
    return_batches: Literal[False] = False,
    batch_size: int | None = None,
    rename: dict[str, str] | Callable | None = None,
    schema_overrides: dict[str, pl.DataType] | None = None,
    execute_options: dict[str, Any] | None = None,
) -> ArrowTable: ...


@overload
def sql_to_df(
    stmt: SQLQuery,
    connection: Engine | AdbcConnection,
    return_type: Literal[DataFrameType.POLARS],
    *,
    return_batches: Literal[True],
    batch_size: int | None = None,
    rename: dict[str, str] | Callable | None = None,
    schema_overrides: dict[str, pl.DataType] | None = None,
    execute_options: dict[str, Any] | None = None,
) -> Iterator[PolarsDataFrame]: ...


@overload
def sql_to_df(
    stmt: SQLQuery,
    connection: Engine | AdbcConnection,
    return_type: Literal[DataFrameType.PANDAS],
    *,
    return_batches: Literal[True],
    batch_size: int | None = None,
    rename: dict[str, str] | Callable | None = None,
    schema_overrides: dict[str, pl.DataType] | None = None,
    execute_options: dict[str, Any] | None = None,
) -> Iterator[PandasDataFrame]: ...


@overload
def sql_to_df(
    stmt: SQLQuery,
    connection: Engine | AdbcConnection,
    return_type: Literal[DataFrameType.ARROW],
    *,
    return_batches: Literal[True],
    batch_size: int | None = None,
    rename: dict[str, str] | Callable | None = None,
    schema_overrides: dict[str, pl.DataType] | None = None,
    execute_options: dict[str, Any] | None = None,
) -> Iterator[ArrowTable]: ...


def sql_to_df(
    stmt: SQLQuery,
    connection: Engine | AdbcConnection,
    return_type: DataFrameType = DataFrameType.PANDAS,
    *,
    return_batches: bool = False,
    batch_size: int | None = None,
    rename: dict[str, str] | Callable | None = None,
    schema_overrides: dict[str, pl.DataType] | None = None,
    execute_options: dict[str, Any] | None = None,
) -> DataFrameClass | Iterator[DataFrameClass]:
    """Execute a SQL string with Polars, returning the requested dataframe type.

    Args:
        stmt: A SQL string to execute.
        connection: A SQLAlchemy Engine or ADBC connection.
        return_type: The dataframe type to return. One of "arrow", "pandas",
            or "polars".
        return_batches: If True, return an iterator that yields each batch
            separately. If False, return a single dataframe with all results.
        batch_size: The size of each batch, when processing in batches.
        rename: A dictionary mapping old column names to new column names, or
            a callable that takes a dataframe and returns a dataframe with
            renamed columns.
        schema_overrides: A dictionary mapping column names to dtypes.
        execute_options: Passed through to the underlying query execution
            method as kwargs.

    Returns:
        If return_batches is False: A dataframe of the query results in the specified
            format.
        If return_batches is True: An iterator of dataframes in the specified format.

    Raises:
        ValueError: If `batch_size` and `return_batches` aren't set together.
    """
    if not batch_size and return_batches:
        raise ValueError("A batch size must be specified if return_batches is True")

    if batch_size and not return_batches:
        raise ValueError("Cannot set a batch size if return_batches is False")

    def _to_format(results: PolarsDataFrame) -> DataFrameClass:
        """Rename the results' columns, then convert to the specified format."""
        if rename:
            results = results.rename(rename)

        return to_dataframe(results, return_type)

    res = pl.read_database(
        query=stmt,
        connection=connection,
        iter_batches=bool(batch_size),
        batch_size=batch_size,
        schema_overrides=schema_overrides,
        execute_options=execute_options,
    )

    if return_batches:
        return (_to_format(batch) for batch in res)

    return _to_format(res)
