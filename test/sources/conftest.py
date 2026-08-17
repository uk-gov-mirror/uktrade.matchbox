"""Warehouse fixtures.

Every test in this directory runs against SQLite, over either a SQLAlchemy engine or
an ADBC connection. None of them needs Docker.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from adbc_driver_sqlite import dbapi as adbc_sqlite
from sqlalchemy import Engine, create_engine


@pytest.fixture
def sqlite_path(tmp_path: Path) -> Path:
    """One SQLite file per test, shared by every client that opens it.

    File-backed rather than `:memory:`, so a test can write through one client and
    read through another. That's the point of parametrising over client types.
    Two `:memory:` connections are two different databases.
    """
    return tmp_path / "warehouse.sqlite"


@pytest.fixture
def sqla_sqlite_warehouse(sqlite_path: Path) -> Iterator[Engine]:
    """A SQLite warehouse, over SQLAlchemy."""
    engine = create_engine(f"sqlite:///{sqlite_path}")
    yield engine
    engine.dispose()


@pytest.fixture
def adbc_sqlite_warehouse(sqlite_path: Path) -> Iterator[adbc_sqlite.Connection]:
    """The same SQLite warehouse, over ADBC."""
    connection = adbc_sqlite.connect(str(sqlite_path))
    yield connection
    connection.close()
