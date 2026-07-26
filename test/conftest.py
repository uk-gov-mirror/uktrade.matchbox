import logging
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console
from sqlalchemy import Engine, create_engine

TEST_ROOT = Path(__file__).resolve().parent


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
    return TEST_ROOT


@pytest.fixture(scope="session", autouse=True)
def patch_rich_console() -> Iterator[None]:
    """Patch Rich console for quiet output in tests.

    A quiet console also keeps `collect`'s progress tree out of the test output;
    `test_progress` swaps in a recording console where it wants to read what was drawn.
    """
    with patch("matchlab.core.logging.console", new=Console(quiet=True)):
        yield
