"""DuckDB engine specifics, what only a real DuckDB file (or `:memory:`) can show.

The backend-agnostic storage contract lives in `test_contract.py`. What stays here is
DuckDB's own file semantics: reclaiming space to disk, the difference between resident
and on-disk bytes, the high-water mark, and the `:memory:` prune hazard. None of that
fits in a contract written against the `Store` ABC.
"""

from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest

from matchlab.stores import DuckDBStore, DuckDBStoreStats

from .conftest import Fingerprints


@pytest.fixture
def memory_store() -> Iterator[DuckDBStore]:
    """An ephemeral in-memory DuckDB store, built directly for the engine tests."""
    store = DuckDBStore(":memory:")
    yield store
    store.close()


# -- resident vs on-disk bytes --------------------------------------------------------


def test_stats_resident_vs_on_disk() -> None:
    """The subclass hook. `4.0 KB` reads as on disk, but not for `:memory:`."""
    resident = DuckDBStoreStats(location=":memory:", bytes=4096)
    on_disk = DuckDBStoreStats(location="/s.duckdb", bytes=4096, path=Path("/s.duckdb"))

    assert resident.describe() == "Store 4.0 KB in memory, 0 artifacts"
    assert on_disk.describe() == "Store 4.0 KB, 0 artifacts"


def test_stats_in_memory_size(
    memory_store: DuckDBStore,
    fp: Fingerprints,
    extract: pl.DataFrame,
    leaves: pl.DataFrame,
) -> None:
    """The regression guard for the in-memory branch.

    An in-memory store allocates no blocks, so anything reading DuckDB's block count
    reports `0 B`. That is true for every test in this suite, and for every
    `DuckDBStore(":memory:")` a user writes. It has no file either, so there is
    nothing to `stat`.
    """
    memory_store.store_source(fp.src, "key", extract, leaves)

    stats = memory_store.stats()

    assert stats.path is None
    assert stats.bytes > 0
    assert stats.free_bytes == 0


# -- on-disk byte accounting ----------------------------------------------------------


def test_stats_on_disk_size(
    tmp_path: Path, fp: Fingerprints, resolver_output: pl.DataFrame
) -> None:
    """The figure has to survive the user checking it with `du`.

    This is measured while the connection is open, which is when a collect reports it.
    Writes sit in the write-ahead log until they are settled into blocks, so a store
    measured before that reads orders of magnitude low.
    """
    db = tmp_path / "store.duckdb"
    store = DuckDBStore(db)
    try:
        store.store_resolver(fp.resolver, resolver_output)
        stats = store.stats()
        on_disk = sum(f.stat().st_size for f in tmp_path.iterdir())

        assert stats.path == db.resolve()
        assert stats.bytes == on_disk
    finally:
        store.close()

    # The on-disk size does not move once the store is closed and reopened.
    assert sum(f.stat().st_size for f in tmp_path.iterdir()) == stats.bytes


def test_stats_grows_with_data(tmp_path: Path, fp: Fingerprints) -> None:
    """Storing more data grows the reported on-disk size."""
    store = DuckDBStore(tmp_path / "store.duckdb")
    try:
        empty = store.stats().bytes
        store.store_transform(
            fp.record_step,
            pl.DataFrame({"id": range(50_000), "name": ["x"] * 50_000}),
        )
        assert store.stats().bytes > empty
    finally:
        store.close()


# -- prune: DuckDB file hazards -------------------------------------------------------


def test_prune_reclaims_disk_space(
    tmp_path: Path, fp: Fingerprints, extract: pl.DataFrame, leaves: pl.DataFrame
) -> None:
    """The assertion the old `gc()` would have failed.

    Purging alone frees nothing. DuckDB reuses freed blocks but never hands them back,
    so the file stays at its high-water mark until it is rewritten.
    """
    store = DuckDBStore(tmp_path / "store.duckdb")
    try:
        store.store_source(fp.src, "key", extract, leaves)
        store.store_transform(
            fp.record_step,
            pl.DataFrame({"id": range(200_000), "name": ["padding"] * 200_000}),
        )
        before = store.stats().bytes

        result = store.prune(keep=[fp.src])

        assert store.stats().bytes < before
        assert result.reclaimed > 0
    finally:
        store.close()


def test_prune_in_memory_keeps_data(
    memory_store: DuckDBStore,
    fp: Fingerprints,
    extract: pl.DataFrame,
    leaves: pl.DataFrame,
    edges: pl.DataFrame,
) -> None:
    """An in-memory store must not be rewritten.

    Reopening `:memory:` hands back an empty database rather than a smaller one, so a
    close-and-swap would destroy the store it was asked to tidy. `:memory:` is what the
    whole suite, and both examples, run on.
    """
    memory_store.store_source(fp.src, "key", extract, leaves)
    memory_store.store_model(fp.model, edges)

    result = memory_store.prune(keep=[fp.src])

    assert result.removed == 1
    assert memory_store.has(fp.src)
    assert memory_store.read_source_extract(fp.src).height == extract.height
