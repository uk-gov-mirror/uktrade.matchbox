"""Base classes for resolver methodologies."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import ClassVar

import polars as pl
from pydantic import BaseModel, ConfigDict

from matchlab.core.config import ResolverType


class ResolverSettings(BaseModel, ABC):
    """Base settings type for resolver methodologies.

    Settings that point at one of the resolver's inputs refer to it by **position**.
    Resolver` accepts the `Model` object at the API and translates it,
    so nothing here holds a plan node: a methodology stays a pure function of edges
    and numbers.
    """

    model_config = ConfigDict(extra="forbid")


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
