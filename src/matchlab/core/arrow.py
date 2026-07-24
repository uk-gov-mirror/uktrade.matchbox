"""Common Arrow utilities."""

from enum import StrEnum
from io import BytesIO
from typing import Final

import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import Schema

from matchlab.core.exceptions import SchemaMismatch

SCHEMA_QUERY: Final[pa.Schema] = pa.schema(
    [("id", pa.int64()), ("key", pa.large_string())]
)
"""Data transfer schema for root cluster IDs keyed to primary keys."""

SCHEMA_QUERY_WITH_LEAVES = SCHEMA_QUERY.append(pa.field("leaf_id", pa.int64()))
"""Data transfer schema for root cluster IDs keyed to primary keys and leaf IDs."""


SCHEMA_INDEX: Final[pa.Schema] = pa.schema(
    [("hash", pa.large_binary()), ("keys", pa.large_list(pa.large_string()))]
)
"""Data transfer schema for data to be indexed."""

SCHEMA_MODEL_EDGES: Final[pa.Schema] = pa.schema(
    [
        ("left_id", pa.uint64()),
        ("right_id", pa.uint64()),
        ("score", pa.float32()),
    ]
)
"""Data transfer schema for the results of a deduplication or linking process."""

SCHEMA_CLUSTERS: Final[pa.Schema] = pa.schema(
    [
        ("parent_id", pa.uint64()),
        ("child_id", pa.uint64()),
    ]
)
"""Data transfer schema for resolver cluster assignments."""


SCHEMA_JUDGEMENTS: Final[pa.Schema] = pa.schema(
    [
        ("user_name", pa.large_string()),
        ("endorsed", pa.uint64()),
        ("shown", pa.uint64()),
    ]
)
"""Data transfer schema for retrieved evaluation judgements from users."""

SCHEMA_CLUSTER_EXPANSION: Final[pa.Schema] = pa.schema(
    [
        ("root", pa.uint64()),
        ("leaves", pa.list_(pa.uint64())),
    ]
)
"""Data transfer schema for mapping from a cluster ID to all its source cluster IDs."""

SCHEMA_EVAL_SAMPLES: Final[pa.Schema] = pa.schema(
    [
        ("root", pa.uint64()),
        ("leaf", pa.uint64()),
        ("key", pa.large_string()),
        ("source", pa.large_string()),
    ]
)
"""Data transfer schema for evaluation samples."""


class JudgementsZipFilenames(StrEnum):
    """Enumeration of file names in ZIP file with downloaded judgements."""

    JUDGEMENTS = "judgements.parquet"
    EXPANSION = "expansion.parquet"


def table_to_buffer(table: pa.Table) -> BytesIO:
    """Converts an Arrow table to a BytesIO buffer."""
    sink = BytesIO()
    pq.write_table(table, sink)
    sink.seek(0)
    return sink


def check_schema_subset(expected: Schema, actual: Schema) -> None:
    """Check presence of Arrow fields, ignoring field order, extras and metadata.

    Args:
        expected: Schema with fields that must be present
        actual: Schema to check
    """
    actual = pa.schema(
        [
            (name, actual.field(name).type)
            for name in expected.names
            if name in actual.names
        ]
    )

    if not actual.equals(expected):
        raise SchemaMismatch(expected=expected, actual=actual)
