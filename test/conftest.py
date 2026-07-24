import logging
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console
from rich.progress import Progress
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
    """Patch Rich console for quiet output in tests."""
    quiet_console = Console(quiet=True)

    console_patch = patch("matchlab.core.logging.console", new=quiet_console)
    progress_patch = patch(
        "matchlab.core.logging.build_progress_bar",
        return_value=Progress(console=quiet_console),
    )

    with console_patch, progress_patch:
        yield
