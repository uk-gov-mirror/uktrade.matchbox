"""Fixtures for the store tests.

The contract tests (`test_contract.py`) run one test body over every backend. A test
that takes the `store` fixture is parametrised across each store implementation. A test
that takes `durable_store` runs only over backends that survive a reopen. Adding a
backend means adding an entry to the lists below, not writing a new test. That is the
same move `DEDUPERS` makes for models, and the `warehouse` dispatch makes for warehouse
clients.

DuckDB is the only backend today. Some behaviour only a real DuckDB *file* can show:
reclaiming space, resident-vs-on-disk bytes, the in-memory-prune hazard. That behaviour
is engine-specific, so it lives in `test_duckdb.py`, which builds its stores directly.
"""

from collections.abc import Callable, Iterator
from pathlib import Path

import polars as pl
import pytest
from pydantic import BaseModel, ConfigDict

from matchlab.stores import DuckDBStore, Store

# Every store, and the subset of them that persists across a reopen.
STORE_BACKENDS = ["duckdb_memory"]
DURABLE_BACKENDS = ["duckdb"]


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrise `store`/`durable_store` over their backends, indirectly."""
    if "store" in metafunc.fixturenames:
        metafunc.parametrize("store", STORE_BACKENDS, indirect=True)
    if "durable_store" in metafunc.fixturenames:
        metafunc.parametrize("durable_store", DURABLE_BACKENDS, indirect=True)


# -- stores ---------------------------------------------------------------------------


@pytest.fixture
def duckdb_memory_store() -> Iterator[Store]:
    """An ephemeral in-memory DuckDB store."""
    store = DuckDBStore(":memory:")
    yield store
    store.close()


@pytest.fixture
def store(request: pytest.FixtureRequest) -> Store:
    """Dispatch to a store by name, for indirect parametrisation."""
    return request.getfixturevalue(f"{request.param}_store")


@pytest.fixture
def duckdb_durable_store(tmp_path: Path) -> Callable[[], Store]:
    """A factory that reopens a file-backed DuckDB store at the same path each call."""
    db = tmp_path / "store.duckdb"
    return lambda: DuckDBStore(db)


@pytest.fixture
def durable_store(request: pytest.FixtureRequest) -> Callable[[], Store]:
    """Dispatch to a durable backend, returning a factory that reopens its store."""
    return request.getfixturevalue(f"{request.param}_durable_store")


# -- artifacts under test -------------------------------------------------------------


class Fingerprints(BaseModel):
    """Distinct 32-byte fingerprints, one per artifact under test."""

    model_config = ConfigDict(frozen=True)

    src: bytes = b"\x01" * 32
    model: bytes = b"\x02" * 32
    resolver: bytes = b"\x03" * 32
    resolver_b: bytes = b"\x0c" * 32
    record_step: bytes = b"\x04" * 32


@pytest.fixture
def fp() -> Fingerprints:
    """The distinct fingerprints the round-trip tests store artifacts under."""
    return Fingerprints()


@pytest.fixture
def extract() -> pl.DataFrame:
    """A source's extract, three records over two columns and a key."""
    return pl.DataFrame(
        {
            "company_name": ["acme", "acme ltd", "beta"],
            "postcode": ["AB1", "AB1", "CD2"],
            "key": ["k1", "k2", "k3"],
        }
    )


@pytest.fixture
def leaves() -> pl.DataFrame:
    """The leaf assignment for `extract`, one leaf per record."""
    return pl.DataFrame(
        {"key": ["k1", "k2", "k3"], "leaf": [1, 2, 3]},
        schema={"key": pl.Utf8, "leaf": pl.UInt64},
    )


@pytest.fixture
def edges() -> pl.DataFrame:
    """A model's scored edges over the source's leaves."""
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
