import importlib.util
import logging
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console
from rich.progress import Progress


# The server-backed fixtures (Postgres/redis/S3/FastAPI) are only usable when the
# `server` extra is installed AND a server environment is configured. Locally — where
# matchlab runs without a server — importing `matchbox.server` fails (it initialises
# from required env vars), so we skip loading those fixtures rather than crash at
# collection. This lets the local suites (adapters, resolution, integration) run with a
# plain `pytest`. In CI, where the server is configured, the fixtures load as before.
# TODO(phase-4): delete these fixtures with matchbox.server and simplify this back.
def _server_fixtures_available() -> bool:
    if importlib.util.find_spec("redis") is None:
        return False
    try:
        import matchbox.server  # noqa: F401, PLC0415 - lazy: triggers server init
    except Exception:  # noqa: BLE001 - any init failure means "no server here"
        return False
    return True


pytest_plugins = (
    [
        "test.fixtures.db",
        "test.fixtures.client",
    ]
    if _server_fixtures_available()
    else []
)

TEST_ROOT = Path(__file__).resolve().parent


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

    console_patch = patch("matchbox.common.logging.console", new=quiet_console)
    progress_patch = patch(
        "matchbox.common.logging.build_progress_bar",
        return_value=Progress(console=quiet_console),
    )

    with console_patch, progress_patch:
        yield
