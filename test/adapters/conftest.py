"""Fixtures for the storage-adapter tests.

The contract tests (`test_contract.py`) run one body over every backend: a test that
takes the `store` fixture is parametrised across each storage backend, and one that
takes `durable_store` across each backend that survives a reopen. The wiring in
`pytest_generate_tests` means a second backend is a new entry in the lists below, not a
new test body — the same move `DEDUPERS` makes for models, and the `warehouse` dispatch
makes for warehouse clients.

DuckDB is the only backend today. Behaviour only a DuckDB *file* can show — reclaiming
space, resident-vs-on-disk bytes, the in-memory-trim hazard — is engine-specific and
lives in `test_duckdb.py`, which builds its stores directly.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest

from matchlab.adapters import Adapter, DuckDBAdapter

# Every storage backend, and the subset of them that persists across a reopen.
STORE_BACKENDS = ["duckdb_memory"]
DURABLE_BACKENDS = ["duckdb"]


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrise `store`/`durable_store` over their backends, indirectly."""
    if "store" in metafunc.fixturenames:
        metafunc.parametrize("store", STORE_BACKENDS, indirect=True)
    if "durable_store" in metafunc.fixturenames:
        metafunc.parametrize("durable_store", DURABLE_BACKENDS, indirect=True)


# -- storage backends -----------------------------------------------------------------


@pytest.fixture
def duckdb_memory_store() -> Iterator[Adapter]:
    """An ephemeral in-memory DuckDB store."""
    store = DuckDBAdapter(":memory:")
    yield store
    store.close()


@pytest.fixture
def store(request: pytest.FixtureRequest) -> Adapter:
    """Dispatch to a storage backend by name, for indirect parametrisation."""
    return request.getfixturevalue(f"{request.param}_store")


@pytest.fixture
def duckdb_durable_store(tmp_path: Path) -> Callable[[], Adapter]:
    """A factory that opens — and reopens — a file-backed DuckDB store at one path."""
    db = tmp_path / "store.duckdb"
    return lambda: DuckDBAdapter(db)


@pytest.fixture
def durable_store(request: pytest.FixtureRequest) -> Callable[[], Adapter]:
    """Dispatch to a durable backend: a factory that reopens the same store."""
    return request.getfixturevalue(f"{request.param}_durable_store")


# -- artifacts under test -------------------------------------------------------------


@dataclass(frozen=True)
class Fingerprints:
    """Distinct 32-byte fingerprints, one per artifact under test."""

    src: bytes = b"\x01" * 32
    model: bytes = b"\x02" * 32
    resolver: bytes = b"\x03" * 32
    resolver_b: bytes = b"\x0c" * 32
    view: bytes = b"\x04" * 32


@pytest.fixture
def fp() -> Fingerprints:
    return Fingerprints()


@pytest.fixture
def extract() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "company_name": ["acme", "acme ltd", "beta"],
            "postcode": ["AB1", "AB1", "CD2"],
            "key": ["k1", "k2", "k3"],
        }
    )


@pytest.fixture
def leaves() -> pl.DataFrame:
    return pl.DataFrame(
        {"key": ["k1", "k2", "k3"], "leaf": [1, 2, 3]},
        schema={"key": pl.Utf8, "leaf": pl.UInt64},
    )


@pytest.fixture
def edges() -> pl.DataFrame:
    return pl.DataFrame(
        {"left_id": [1, 3], "right_id": [2, 4], "score": [0.9, 0.8]},
        schema={"left_id": pl.UInt64, "right_id": pl.UInt64, "score": pl.Float32},
    )


@pytest.fixture
def resolver_output() -> pl.DataFrame:
    """root/leaf/key/source == SCHEMA_RESOLVER_OUTPUT, over two sources."""
    return pl.DataFrame(
        {
            "root": [10, 10, 20, 20],
            "leaf": [1, 2, 3, 4],
            "key": ["k1", "k2", "k3", "k4"],
            "source": ["crn", "crn", "dh", "dh"],
        },
        schema={
            "root": pl.UInt64,
            "leaf": pl.UInt64,
            "key": pl.Utf8,
            "source": pl.Utf8,
        },
    )
