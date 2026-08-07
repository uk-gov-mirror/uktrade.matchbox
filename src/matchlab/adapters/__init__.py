"""Storage adapters for matchlab.

An adapter persists collected DAG-step artifacts keyed by fingerprint. It is storage,
not an engine that resolves on demand. `DuckDBAdapter` is the reference implementation.
"""

from matchlab.adapters.base import (
    Adapter,
    Fingerprint,
    StoreStats,
    TrimResult,
    format_bytes,
)
from matchlab.adapters.duckdb import DuckDBAdapter, DuckDBStoreStats

__all__ = (
    "Adapter",
    "DuckDBAdapter",
    "DuckDBStoreStats",
    "Fingerprint",
    "StoreStats",
    "TrimResult",
    "format_bytes",
)
