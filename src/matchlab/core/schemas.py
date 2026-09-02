"""Arrow schemas for the artifacts a store persists.

These are the shapes that cross the boundary between a step and its storage: what a
`Model` or `Resolver` hands the store, and what the store hands back. They live here
rather than with their producers because both sides need them, and `core` is the only
layer both a producer and a store can depend on without inverting.

The membership rule is exactly that boundary — **the store persists it or returns
it**. A shape that never leaves the process belongs with the code that produces it:
`SCHEMA_CLUSTERS` is a resolver methodology's output, read by the `Resolver` that called
it and by nothing else, so it lives in `matchlab.resolvers.base`. Keeping those out is
what stops this module from becoming a catch-all for anything that happens to be an
Arrow schema.
"""

from typing import Final

import pyarrow as pa
from pyarrow import Schema

from matchlab.core.exceptions import SchemaMismatch

SCHEMA_MODEL_EDGES: Final[pa.Schema] = pa.schema(
    [
        ("left_id", pa.uint64()),
        ("right_id", pa.uint64()),
        ("score", pa.float32()),
    ]
)
"""A deduper's or linker's output: a scored edge list."""

SCHEMA_RESOLVER_OUTPUT: Final[pa.Schema] = pa.schema(
    [
        ("root", pa.uint64()),
        ("leaf", pa.uint64()),
        ("key", pa.large_string()),
        ("source", pa.large_string()),
    ]
)
"""A resolver's complete flat output: one row per reachable source record.

Evaluation sampling returns a row-subset of this, which is why it is named for the
resolver's output rather than for the sample.
"""

SCHEMA_JUDGEMENTS: Final[pa.Schema] = pa.schema(
    [
        ("user_name", pa.large_string()),
        ("endorsed", pa.uint64()),
        ("shown", pa.uint64()),
    ]
)
"""Stored user judgements, as returned by `Store.read_eval_data`."""

SCHEMA_CLUSTER_EXPANSION: Final[pa.Schema] = pa.schema(
    [
        ("root", pa.uint64()),
        ("leaves", pa.list_(pa.uint64())),
    ]
)
"""A cluster ID mapped to all its source cluster IDs. 

Used alongside `SCHEMA_JUDGEMENTS` in evaluation scoring."""


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
