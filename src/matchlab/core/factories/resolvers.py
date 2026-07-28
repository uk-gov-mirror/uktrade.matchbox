"""Factory helpers for resolver testkits."""

from collections.abc import Hashable, Iterable, Mapping
from typing import Annotated, Any, ClassVar, TypeVar

import polars as pl
from faker import Faker
from pydantic import BaseModel, ConfigDict, Field

from matchlab.core.dsu import DisjointSet
from matchlab.core.factories.entities import ClusterEntity, SourceEntity
from matchlab.core.factories.models import ModelTestkit, model_factory
from matchlab.core.factories.sources import linked_sources_factory
from matchlab.models import Model
from matchlab.resolvers import (
    Resolver,
    ResolverMethod,
    ResolverSettings,
    ResolverType,
    add_resolver_class,
)
from matchlab.resolvers.base import SCHEMA_CLUSTERS
from matchlab.sources import Source
from matchlab.views import View

K = TypeVar("K", bound=Hashable)


class MockResolverSettings(ResolverSettings):
    """Settings type for MockResolver."""

    thresholds: dict[int, Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        default_factory=dict
    )


def _connected_components_from_edges(
    model_edges: Mapping[K, pl.DataFrame],
    thresholds: Mapping[K, float],
) -> pl.DataFrame:
    """Generate clusters from model edge tables and per-model thresholds.

    Keyed by whatever identifies an input to the caller: positions in a plan, names in
    the factory's own bookkeeping.
    """
    djs = DisjointSet[int]()

    for key, edges in model_edges.items():
        if edges.height == 0:
            continue

        threshold = thresholds[key]
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


class MockResolver(ResolverMethod):
    """Mock resolver methodology used by resolver testkits."""

    resolver_type: ClassVar[ResolverType] = ResolverType.COMPONENTS
    settings: MockResolverSettings

    def compute_clusters(self, model_edges: Mapping[int, pl.DataFrame]) -> pl.DataFrame:
        """Compute mock clusters with connected components."""
        return _connected_components_from_edges(
            model_edges=model_edges,
            thresholds={
                position: self.settings.thresholds.get(position, 0.0)
                for position in model_edges
            },
        )


add_resolver_class(MockResolver)


class ResolverTestkit(BaseModel):
    """Resolver plus local expected data for tests."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    resolver: Resolver
    #: The testkit's own name, for bookkeeping between fixtures — not a published one.
    name: str
    assignments: pl.DataFrame
    entities: tuple[ClusterEntity, ...]

    def view(self, *args: Any, **kwargs: Any) -> View:
        """Thin wrapper to build a cleaned view through this testkit's Resolve."""
        return self.resolver.view(*args, **kwargs)


def resolver_factory(
    inputs: Iterable[ModelTestkit] | None = None,
    true_entities: Iterable[SourceEntity] | None = None,
    name: str | None = None,
    thresholds: Mapping[str, float] | None = None,
    seed: int = 42,
) -> ResolverTestkit:
    """Generate a complete resolver testkit.

    Allows autoconfiguration with minimal settings, or more nuanced control.

    Can either be used to generate a resolver in a pipeline, interconnected with
    existing testkit objects, or generate a standalone resolver with random data.

    Args:
        inputs: An iterable of ModelTestkit objects to use as resolver inputs.
            If None, a single default deduper model testkit is created automatically.
        true_entities: Ground truth SourceEntity objects used to generate the
            expected cluster assignments. If None, the resolver testkit will have
            no expected entities.
        name: Name for the testkit, used to refer to it between fixtures. Defaults to
            a randomly generated word suffixed with '_resolver'.
        thresholds: Per-model score thresholds in [0.0, 1.0]. If omitted,
            defaults to 0.0 for all resolver inputs.
        seed: Random seed for reproducibility.

    Returns:
        ResolverTestkit: A resolver testkit with generated assignments and expected
            entities.

    Raises:
        TypeError: If any element of inputs is not a ModelTestkit.
    """
    if inputs is None:
        linked = linked_sources_factory(seed=seed)
        default_model = model_factory(
            left_testkit=linked.sources["crn"],
            true_entities=tuple(linked.true_entities),
            seed=seed,
        )
        inputs = (default_model,)
        source_names: set[str] = set(linked.sources.keys())
    else:
        inputs = tuple(inputs)
        source_names: set[str] = set()
        for testkit in inputs:
            if not isinstance(testkit, ModelTestkit):
                raise TypeError("resolver_factory inputs must be ModelTestkit.")
            source_names.update(
                step.name
                for step in testkit.model.lineage()
                if isinstance(step, Source)
            )

    input_map: dict[str, ModelTestkit] = {}
    for testkit in inputs:
        input_map.setdefault(testkit.name, testkit)

    resolver_inputs: list[Model] = []
    model_edges: dict[str, pl.DataFrame] = {}
    for testkit in input_map.values():
        resolver_inputs.append(testkit.model)
        model_edges[testkit.name] = testkit.scores

    expected_model_names = set(input_map.keys())
    if thresholds is None:
        thresholds = {name: 0.0 for name in expected_model_names}

    if set(thresholds) != expected_model_names:
        raise ValueError("Threshold keys must exactly match resolver input models.")

    # Hand `Resolver` the model objects and let it work out the positions, so the
    # factory exercises the same translation every hand-written plan goes through.
    resolver_settings = {
        "thresholds": {
            input_map[model_name].model: threshold
            for model_name, threshold in thresholds.items()
        }
    }

    generator = Faker()
    generator.seed_instance(seed)
    resolver = Resolver(
        *resolver_inputs,
        resolver_class=MockResolver,
        resolver_settings=resolver_settings,
    )
    assignments = _connected_components_from_edges(
        model_edges=model_edges,
        thresholds=thresholds,
    )

    source_entities = tuple(true_entities or ())
    entities = tuple(
        projected
        for entity in source_entities
        if (projected := entity.to_cluster_entity(*source_names)) is not None
    )

    return ResolverTestkit(
        resolver=resolver,
        name=name or f"{generator.unique.word()}_resolver",
        assignments=assignments,
        entities=entities,
    )
