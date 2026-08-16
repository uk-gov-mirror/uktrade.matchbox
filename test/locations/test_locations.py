"""Locations: the one contract, and what is genuinely specific to each kind.

A `Location` has exactly one abstract method, `read`, with four knobs. Every location
implements all four *separately* — a driver batch or a slice, `sql_to_df(rename=)` or
`frame.rename()` — so the contract, not the implementation, is what these are organised
around. The `location` fixture supplies each kind returning identical rows, and the
contract tests below run against all of them.

What is left specific to one kind is a small minority, and says so. Fixtures live in
`conftest.py`; none of this needs Docker.
"""

from typing import Any, cast

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
from adbc_driver_manager import ProgrammingError
from adbc_driver_manager.dbapi import Connection as AdbcConnection
from pydantic import ValidationError
from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError

from matchlab.core.dataframes import DataFrameType
from matchlab.locations import (
    ClientType,
    DataFrame,
    Location,
    RelationalDB,
    add_location_class,
    resolve_location_class,
)

ROWS = pl.DataFrame(
    {
        "key": ["1", "2", "3", "4"],
        "company": ["acme", "beta", "gamma", "delta"],
        "employees": [10, 20, 30, 40],
    }
)

QUERY = "select key, company, employees from rows"


@pytest.fixture
def relational_sqla(sqla_sqlite_warehouse: Engine) -> RelationalDB:
    """`ROWS` in SQLite, read over SQLAlchemy."""
    ROWS.write_database(
        "rows", connection=sqla_sqlite_warehouse, if_table_exists="replace"
    )
    return RelationalDB(sql=QUERY, client=sqla_sqlite_warehouse)


@pytest.fixture
def relational_adbc(
    sqla_sqlite_warehouse: Engine, adbc_sqlite_warehouse: AdbcConnection
) -> RelationalDB:
    """The same rows in the same file, read over ADBC.

    Written through SQLAlchemy because polars writes that way, and read back through
    ADBC: the point of the `sqlite_path` fixture being file-backed.
    """
    ROWS.write_database(
        "rows", connection=sqla_sqlite_warehouse, if_table_exists="replace"
    )
    return RelationalDB(sql=QUERY, client=adbc_sqlite_warehouse)


@pytest.fixture
def dataframe() -> DataFrame:
    """The same rows, already in memory."""
    return DataFrame(df=ROWS)


@pytest.fixture(params=["relational_sqla", "relational_adbc", "dataframe"])
def location(request: pytest.FixtureRequest) -> Location:
    """Every kind of location, each returning `ROWS`."""
    return request.getfixturevalue(request.param)


def read_batches(location: Location, **kwargs: Any) -> list[pl.DataFrame]:
    """The batches a location returns, as polars."""
    return cast("list[pl.DataFrame]", list(location.read(**kwargs)))


def read_all(location: Location, **kwargs: Any) -> pl.DataFrame:
    """Every row a location returns, as one polars frame."""
    return pl.concat(read_batches(location, **kwargs))


def _ignores_batch_size(location: Location) -> bool:
    """Whether this location silently returns everything in one batch.

    Polars registers ADBC with `exact_batch_size: False`, so it calls
    `fetch_record_batch()` with no size and the driver chunks how it likes. See
    TODO(adbc-batching) in `locations.py`. This is the shape of that bug, not a gap.
    """
    return isinstance(location, RelationalDB) and (
        location.client_type is ClientType.ADBC
    )


# -- the read contract, honoured by every location ------------------------------------


def test_read_yields_every_row(location: Location) -> None:
    """A read returns the location's rows, whole, as polars by default."""
    combined = read_all(location)

    assert combined.height == 4
    assert set(combined.columns) == {"key", "company", "employees"}
    assert set(combined["company"]) == {"acme", "beta", "gamma", "delta"}


def test_read_batches(location: Location) -> None:
    """`batch_size` bounds each batch, so a large source never lands in memory whole."""
    batches = read_batches(location, batch_size=3)

    assert pl.concat(batches).height == 4
    if not _ignores_batch_size(location):
        assert [batch.height for batch in batches] == [3, 1]


def test_read_applies_rename(location: Location) -> None:
    """Renaming happens at the location, which is what qualifies a source's columns."""
    renamed = read_all(location, rename={"company": "name"})

    assert "name" in renamed.columns
    assert "company" not in renamed.columns


def test_read_applies_schema_overrides(location: Location) -> None:
    """Overrides pin a column's type, rather than letting the location infer it.

    This is how a source's key is read as a string whatever it is stored as.
    """
    overridden = read_all(location, schema_overrides={"employees": pl.String()})

    assert overridden["employees"].dtype == pl.String
    assert read_all(location)["employees"].dtype == pl.Int64


def test_read_returns_requested_type(location: Location) -> None:
    """The caller picks the frame library, and gets it from every kind of location."""
    assert isinstance(next(location.read()), pl.DataFrame)
    assert isinstance(
        next(location.read(return_type=DataFrameType.PANDAS)), pd.DataFrame
    )
    assert isinstance(next(location.read(return_type=DataFrameType.ARROW)), pa.Table)


# -- relational specifics --------------------------------------------------------------


@pytest.mark.parametrize(
    ("warehouse", "expected"),
    [
        pytest.param("sqla_sqlite", ClientType.SQLALCHEMY, id="sqlalchemy"),
        pytest.param("adbc_sqlite", ClientType.ADBC, id="adbc"),
    ],
    indirect=["warehouse"],
)
def test_client_type_reflects_the_client(
    warehouse: Engine | AdbcConnection, expected: ClientType
) -> None:
    """One `isinstance` decides it, because the field type rejected anything else."""
    assert RelationalDB(sql=QUERY, client=warehouse).client_type == expected


@pytest.mark.parametrize("warehouse", ["sqla_sqlite", "adbc_sqlite"], indirect=True)
def test_invalid_query_raises(
    warehouse: Engine | AdbcConnection,
) -> None:
    """Matchlab does not parse your SQL, so a bad query fails as the database says.

    `OperationalError` for SQLAlchemy, `ProgrammingError` for ADBC.
    """
    location = RelationalDB(sql="select * from nonexistent_table", client=warehouse)

    with pytest.raises((OperationalError, ProgrammingError)):
        list(location.read(batch_size=10))


def test_location_is_typed_and_settled(sqla_sqlite_warehouse: Engine) -> None:
    """The field's type is the check, and a location settles at construction."""
    with pytest.raises(ValidationError):
        RelationalDB(sql=QUERY, client=12)

    location = RelationalDB(sql=QUERY, client=sqla_sqlite_warehouse)
    with pytest.raises(ValidationError):
        location.client = sqla_sqlite_warehouse


# -- dataframe specifics ---------------------------------------------------------------


@pytest.mark.parametrize(
    "frame",
    [
        pytest.param(ROWS, id="polars"),
        pytest.param(ROWS.to_pandas(), id="pandas"),
        pytest.param(ROWS.to_arrow(), id="arrow"),
    ],
)
def test_accepts_polars_pandas_and_arrow(
    frame: pl.DataFrame | pd.DataFrame | pa.Table,
) -> None:
    """All three convert to polars on read.

    Only the conversion differs; everything after it is the code the contract tests
    above already cover, so this does not repeat them per frame library.
    """
    combined = read_all(DataFrame(df=frame))

    assert combined.height == 4
    assert set(combined.columns) == {"key", "company", "employees"}


# -- the registry ----------------------------------------------------------------------


def test_location_found_by_name() -> None:
    """Documents name location classes the way they name dedupers and resolvers."""
    assert resolve_location_class("RelationalDB") is RelationalDB
    assert resolve_location_class(RelationalDB) is RelationalDB

    with pytest.raises(ValueError, match="No location class named 'Nowhere'"):
        resolve_location_class("Nowhere")


def test_registry_is_open() -> None:
    """So a document can travel to a codebase we don't ship."""

    class CustomLocation(RelationalDB):
        pass

    with pytest.raises(ValueError, match="not a subclass of Location"):
        add_location_class(int)

    add_location_class(CustomLocation)
    assert resolve_location_class("CustomLocation") is CustomLocation
