"""Shared fixtures.

Two fixture idioms cover most of the suite (see `test/README.md`):

- The `crn`/`dh` scenario below: a small SQLite `warehouse` with known rows, for plan
  and end-to-end tests that assert on specific records.
- The `linked_sources_factory` oracle in `matchlab.testkit`, for methodology
  correctness at scale.

Every collecting test also gets the autouse in-memory `store`, so nothing ever
touches the real store in the user's cache directory.
"""

import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console
from sqlalchemy import Engine, create_engine, text

from matchlab import Resource, Source, read_database
from matchlab.stores import DuckDBStore
from matchlab.stores.default import set_default_store

TEST_ROOT = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def store() -> Iterator[DuckDBStore]:
    """An in-memory store, set as the default `collect()` reaches for.

    Autouse so no test can fall through to `default_store()`, which would create a
    real DuckDB file in the user's cache directory.
    """
    store = DuckDBStore(":memory:")
    set_default_store(store)
    yield store
    set_default_store(None)
    store.close()


@pytest.fixture
def warehouse(tmp_path: Path) -> Engine:
    """The canonical `crn`/`dh` scenario: a dedupe feeding a cross-source link.

        crn: a1=(acme,london) a2=(acme,leeds) a3=(beta,hull)
        dh:  b1=(acme,bristol) b2=(gamma,york)

    a1/a2 differ on town, so they are distinct leaves and a deduper does real work. The
    apex is a *link*, yet the dedupe grouping of {a1,a2} must survive it, and a3/b2 —
    reachable but matched by nothing — must stay singletons (merge-forward). A file that
    needs a different shape (a distinct second column, a deliberately-agreeing town)
    overrides this fixture locally and says why.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'wh.sqlite'}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE crn (pk TEXT, company TEXT, town TEXT)"))
        conn.execute(
            text(
                "INSERT INTO crn VALUES "
                "('a1','acme','london'),('a2','acme','leeds'),('a3','beta','hull')"
            )
        )
        conn.execute(text("CREATE TABLE dh (pk TEXT, company TEXT, town TEXT)"))
        conn.execute(
            text("INSERT INTO dh VALUES ('b1','acme','bristol'),('b2','gamma','york')")
        )
    return engine


# A callable that builds a `Source` over the shared `warehouse` by table name.
SourceFactory = Callable[..., Source]


@pytest.fixture
def source(warehouse: Engine) -> SourceFactory:
    """Build a `Source` reading one of the `warehouse` tables by name.

    The default extract selects `pk, company, town`. Pass `extract=` to read a subset
    or a different shape. This is the one way the suite turns the `warehouse` into a
    `Source`, so tests and plan builders take it rather than hand-rolling a location.

    The client defaults to a `Resource` named `warehouse`. Pass
    `client=` for the tests that vary it: a bare engine, a second warehouse, or one
    deliberately sharing a name with another object.
    """

    def _make(
        name: str,
        extract: str | None = None,
        client: Engine | Resource[Engine] | None = None,
    ) -> Source:
        return read_database(
            name,
            sql=extract or f"select pk, company, town from {name}",
            client=Resource("warehouse", warehouse) if client is None else client,
            key_field="pk",
        )

    return _make


@pytest.fixture
def sqlite_in_memory_warehouse() -> Iterator[Engine]:
    """An in-memory SQLite warehouse, scoped to one test."""
    engine = create_engine("sqlite:///:memory:")
    yield engine
    engine.dispose()


def pytest_configure() -> None:
    """Configure pytest settings."""
    # Quieten down the logging for specific loggers
    logging.getLogger("faker").setLevel(logging.WARNING)


@pytest.fixture(scope="session")
def test_root_dir() -> Path:
    """The `test/` directory, for fixtures that read files alongside the tests."""
    return TEST_ROOT


@pytest.fixture(scope="session", autouse=True)
def patch_rich_console() -> Iterator[None]:
    """Patch Rich console for quiet output in tests.

    A quiet console also keeps `collect`'s progress tree out of the test output;
    `test_progress` swaps in a recording console where it wants to read what was drawn.
    """
    with patch("matchlab.core.logging.console", new=Console(quiet=True)):
        yield
