"""Connected-components resolver methodology."""

from collections.abc import Mapping
from typing import Annotated, ClassVar

import polars as pl
from pydantic import Field

from matchlab.core.dsu import DisjointSet
from matchlab.resolvers.base import (
    SCHEMA_CLUSTERS,
    ResolverMethod,
    ResolverType,
)


class Components(ResolverMethod):
    """Resolver methodology that computes connected components.

    A threshold defaults to 0.0 if not set.
    """

    version: ClassVar[int] = 1

    resolver_type: ClassVar[ResolverType] = ResolverType.COMPONENTS
    thresholds: dict[int, Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        default_factory=dict,
        description=(
            "Minimum score for an edge to count, per input, keyed by the input's "
            "position. Write these as `{model: 0.9}`. `Resolver` takes the model "
            "object and works out the position."
        ),
    )

    def compute_clusters(  # noqa: D102
        self, model_edges: Mapping[int, pl.DataFrame]
    ) -> pl.DataFrame:
        djs = DisjointSet[int]()

        for position, edges in model_edges.items():
            if edges.height == 0:
                continue

            threshold = self.thresholds.get(position, 0.0)
            filtered_edges = edges.filter(pl.col("score") >= threshold)

            for left_id, right_id in filtered_edges.select(
                "left_id", "right_id"
            ).iter_rows():
                djs.union(left_id, right_id)

        rows: list[dict[str, int]] = []
        for parent_id, component in enumerate(djs.get_components(), start=1):
            rows.extend(
                {"parent_id": parent_id, "child_id": node_id} for node_id in component
            )

        if not rows:
            return pl.from_arrow(SCHEMA_CLUSTERS.empty_table())

        return pl.DataFrame(rows).cast(pl.Schema(SCHEMA_CLUSTERS))
