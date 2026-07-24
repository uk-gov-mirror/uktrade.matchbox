"""Connected-components resolver methodology."""

from collections.abc import Iterable, Mapping
from typing import Annotated, ClassVar

import polars as pl
from pydantic import Field

from matchlab.core.arrow import SCHEMA_CLUSTERS
from matchlab.core.config import ModelStepName, ResolverType
from matchlab.core.dsu import DisjointSet
from matchlab.resolvers.base import ResolverMethod, ResolverSettings


class ComponentsSettings(ResolverSettings):
    """Settings for the Components resolver methodology."""

    thresholds: dict[ModelStepName, Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        default_factory=dict
    )

    def validate_inputs(self, model_names: Iterable[ModelStepName]) -> None:  # noqa: D102
        if invalid := set(self.thresholds.keys()) - set(model_names):
            raise RuntimeError(f"Unknown models in thresholds: {invalid}")


class Components(ResolverMethod):
    """Resolver methodology that computes connected components.

    Thresholds are assumed to be 0.0 unless otherwise specified.
    """

    resolver_type: ClassVar[ResolverType] = ResolverType.COMPONENTS
    settings: ComponentsSettings

    def compute_clusters(  # noqa: D102
        self, model_edges: Mapping[ModelStepName, pl.DataFrame]
    ) -> pl.DataFrame:
        self.settings.validate_inputs(model_edges.keys())

        djs = DisjointSet[int]()

        for model_name, edges in model_edges.items():
            if edges.height == 0:
                continue

            threshold = self.settings.thresholds.get(model_name, 0.0)
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
