"""Warehouse fixtures.

These replace the half of the deleted `test/fixtures/db.py` that wasn't about the
server. None of them needs Docker, which is the point: every test in this directory
either runs against in-memory SQLite or only asks a client what SQL dialect it speaks.

`RelationalDBLocation.validate_extract_transform` reads the dialect off the client —
`engine.dialect.name` for SQLAlchemy — and SQLAlchemy builds an engine lazily, without
connecting. So the dialect-comparison tests, which are about sqlglot parsing rather
than about a database, get a real Postgres-dialect `Engine` that never opens a socket.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from adbc_driver_sqlite import dbapi as adbc_sqlite
from sqlalchemy import Engine, create_engine


@pytest.fixture
def sqlite_path(tmp_path: Path) -> Path:
    """One SQLite file per test, shared by every client that opens it.

    File-backed rather than `:memory:` so that a test can write through one client
    and read through another — which is the point of parametrising over client types.
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


@pytest.fixture
def sqla_postgres_warehouse() -> Iterator[Engine]:
    """A Postgres-dialect engine that never connects.

    Only `engine.dialect.name` is read, so this is enough to test that a statement
    valid in Postgres is rejected in SQLite and vice versa. Anything that needs rows
    back should use a SQLite warehouse instead.
    """
    engine = create_engine("postgresql+psycopg://warehouse:warehouse@localhost/unused")
    yield engine
    engine.dispose()


@pytest.fixture
def warehouse(request: pytest.FixtureRequest) -> object:
    """Dispatch to a warehouse client by name, for indirect parametrisation.

    Lets one test body run against several client types:

        @pytest.mark.parametrize("warehouse", ["sqla_sqlite", "adbc_sqlite"],
                                 indirect=True)
    """
    return request.getfixturevalue(f"{request.param}_warehouse")
