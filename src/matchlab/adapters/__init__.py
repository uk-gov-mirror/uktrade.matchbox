"""Storage adapters for matchlab.

An adapter persists collected DAG-step artifacts keyed by fingerprint. It is storage,
not a resolution engine. `DuckDBAdapter` is the reference implementation.
"""

from matchlab.adapters.base import Adapter, Fingerprint
from matchlab.adapters.duckdb import DuckDBAdapter

__all__ = ("Adapter", "DuckDBAdapter", "Fingerprint")
