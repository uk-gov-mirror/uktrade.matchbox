"""Base classes for resolver methodologies."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import StrEnum
from typing import ClassVar, Final

import polars as pl
import pyarrow as pa
from pydantic import BaseModel, ConfigDict

SCHEMA_CLUSTERS: Final[pa.Schema] = pa.schema(
    [
        ("parent_id", pa.uint64()),
        ("child_id", pa.uint64()),
    ]
)
"""A resolver methodology's cluster assignments take this shape."""


class ResolverType(StrEnum):
    """Enumeration of supported resolver methodology types.

    This lives with `ResolverMethod`, not in `matchlab.specs`, because it is not part
    of any spec. A `ResolverSpec` identifies a methodology only by its registered
    `resolver_class`, so a `ResolverType` never reaches a fingerprint or a document.
    It is just a class attribute methodologies use to declare which family they
    belong to, and nothing more.
    """

    COMPONENTS = "components"


class ResolverSettings(BaseModel, ABC):
    """Base settings type for resolver methodologies.

    Settings that point at one of the resolver's inputs refer to it by **position**.
    `Resolver` accepts the `Model` object at the API and translates it. Nothing here
    holds a plan node. A methodology stays a pure function of edges and numbers.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class ResolverMethod(BaseModel, ABC):
    """Base class for resolver methodologies."""

    resolver_type: ClassVar[ResolverType]
    settings: ResolverSettings

    @abstractmethod
    def compute_clusters(self, model_edges: Mapping[int, pl.DataFrame]) -> pl.DataFrame:
        """Compute cluster assignments from model edges.

        Args:
            model_edges: Input position to that model's edges, conforming to
                SCHEMA_MODEL_EDGES. Positions index the resolver's inputs, in the
                order they were given, and are what per-model settings key by.

        Returns:
            A Polars DataFrame which conforms to SCHEMA_CLUSTERS
        """
        ...
