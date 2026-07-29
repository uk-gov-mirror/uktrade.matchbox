"""Warehouse fixtures.

These replace the half of the deleted `test/fixtures/db.py` that wasn't about the
server. None of them needs Docker, which is the point: every test in this directory
either runs against in-memory SQLite or only asks a client what SQL dialect it speaks.

`RelationalDBLocation.validate_extract_transform` reads the dialect off the client —
`engine.dialect.name` for SQLAlchemy. So the dialect-comparison tests, which are about
sqlglot parsing rather than about a database, get a Postgres-dialect `Engine` with
nothing behind it.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import NoReturn

import pytest
from adbc_driver_sqlite import dbapi as adbc_sqlite
from sqlalchemy import Engine, create_engine, make_url
from sqlalchemy.dialects.postgresql.base import PGDialect
from sqlalchemy.pool import NullPool


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
def sqla_postgres_dialect() -> Engine:
    """A Postgres-dialect engine with no driver and no database behind it.

    Deliberately not named `..._warehouse`: it holds no rows and never will, so it is
    not one, and the `warehouse` fixture below cannot reach it. Only
    `engine.dialect.name` is read, which is enough to test that a statement valid in
    Postgres is rejected in SQLite and vice versa.

    Assembled from `PGDialect` rather than via `create_engine`, because `create_engine`
    imports the URL's DBAPI driver — and a driver is a dependency we'd be taking on
    solely to never call it. The pool's creator raises for the same reason: a test that
    starts connecting through this engine should say so, not reach for a database that
    was never there.
    """

    def _no_driver() -> NoReturn:
        raise RuntimeError(
            "sqla_postgres_dialect is a dialect, not a database. Use a SQLite "
            "warehouse for anything that needs rows back."
        )

    return Engine(
        pool=NullPool(creator=_no_driver),
        dialect=PGDialect(),
        url=make_url("postgresql://warehouse/unused"),
    )


@pytest.fixture
def warehouse(request: pytest.FixtureRequest) -> object:
    """Dispatch to a warehouse client by name, for indirect parametrisation.

    Lets one test body run against several client types:

        @pytest.mark.parametrize("warehouse", ["sqla_sqlite", "adbc_sqlite"],
                                 indirect=True)
    """
    return request.getfixturevalue(f"{request.param}_warehouse")
