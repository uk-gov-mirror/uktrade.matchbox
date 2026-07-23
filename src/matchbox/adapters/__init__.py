"""Storage adapters for local-only Matchbox / matchlab.

An adapter persists collected DAG-step artifacts keyed by fingerprint. It is storage,
not a resolution engine. `DuckDBAdapter` is the reference implementation.
"""

from matchbox.adapters.base import Adapter, Fingerprint
from matchbox.adapters.duckdb import DuckDBAdapter

__all__ = ("Adapter", "DuckDBAdapter", "Fingerprint")
