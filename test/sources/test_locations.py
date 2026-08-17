"""Tests for the `Location` read contract.

`Location` has one abstract method, `read`. Each kind of location implements it from
scratch, so the contract tests here run against every kind. The `location` fixture
supplies each one returning the same rows.

The sections after that cover what is true of only one kind.
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
from matchlab.sources import (
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


@pytest.fixture(params=["sqlalchemy", "adbc", "dataframe"])
def location(request: pytest.FixtureRequest) -> Location:
    """Every kind of location, each returning `ROWS`.

    The rows are written through SQLAlchemy because that is how polars writes, then
    read back through whichever client this case names. Reading through one client
    what another wrote is why `sqlite_path` is file-backed.
    """
    if request.param == "dataframe":
        return DataFrame(df=ROWS)

    engine: Engine = request.getfixturevalue("sqla_sqlite_warehouse")
    ROWS.write_database("rows", connection=engine, if_table_exists="replace")
    client = (
        engine
        if request.param == "sqlalchemy"
        else request.getfixturevalue("adbc_sqlite_warehouse")
    )
    return RelationalDB(sql=QUERY, client=client)


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
    TODO(adbc-batching) in `sources/relational.py`. This is the shape of that bug,
    not a gap in the test.
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


def test_client_type_reflects_the_client(
    sqla_sqlite_warehouse: Engine, adbc_sqlite_warehouse: AdbcConnection
) -> None:
    """One `isinstance` decides it, because the field type rejected anything else."""
    sqla = RelationalDB(sql=QUERY, client=sqla_sqlite_warehouse)
    adbc = RelationalDB(sql=QUERY, client=adbc_sqlite_warehouse)

    assert sqla.client_type is ClientType.SQLALCHEMY
    assert adbc.client_type is ClientType.ADBC


@pytest.mark.parametrize(
    "client",
    ["sqla_sqlite_warehouse", "adbc_sqlite_warehouse"],
    ids=["sqlalchemy", "adbc"],
)
def test_invalid_query_raises(request: pytest.FixtureRequest, client: str) -> None:
    """Matchlab does not parse your SQL, so a bad query fails as the database says.

    `OperationalError` for SQLAlchemy, `ProgrammingError` for ADBC.
    """
    location = RelationalDB(
        sql="select * from nonexistent_table", client=request.getfixturevalue(client)
    )

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

    Only the conversion differs. The contract tests above already cover everything
    after it, so this does not repeat them for each frame library.
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
    """A document can then travel to a codebase we don't ship."""

    class CustomLocation(RelationalDB):
        pass

    with pytest.raises(ValueError, match="not a subclass of Location"):
        add_location_class(int)

    add_location_class(CustomLocation)
    assert resolve_location_class("CustomLocation") is CustomLocation
