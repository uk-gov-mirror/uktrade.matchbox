"""Functions and variables for the default matchlib store."""

from platformdirs import user_cache_path

from matchlab.stores.base import Store
from matchlab.stores.duckdb import DuckDBStore

_DEFAULT_STORE: Store | None = None

CACHE_DIR = user_cache_path("matchlab")


def default_store() -> Store:
    """Return the default store, creating a DuckDB store in the cache dir if unset."""
    global _DEFAULT_STORE  # module-level default, by design
    if _DEFAULT_STORE is None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _DEFAULT_STORE = DuckDBStore(CACHE_DIR / "store.duckdb")
    return _DEFAULT_STORE


def set_default_store(store: Store | None) -> None:
    """Set the store used by `collect()` when none is passed. `None` resets it."""
    global _DEFAULT_STORE  # module-level default, by design
    _DEFAULT_STORE = store
