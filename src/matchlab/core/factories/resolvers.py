"""Factory helpers for resolver testkits."""

from collections.abc import Iterable, Mapping
from typing import Annotated, Any, ClassVar

import polars as pl
from faker import Faker
from pydantic import BaseModel, ConfigDict, Field

from matchlab.cleaning import Clean
from matchlab.core.arrow import SCHEMA_CLUSTERS
from matchlab.core.config import (
    ModelStepName,
    ResolverStepName,
    ResolverType,
    SourceStepName,
)
from matchlab.core.factories.entities import ClusterEntity, SourceEntity
from matchlab.core.factories.models import ModelTestkit, model_factory
from matchlab.core.factories.sources import linked_sources_factory
from matchlab.core.transform import DisjointSet
from matchlab.models import Model
from matchlab.resolvers import (
    Resolve,
    ResolverMethod,
    ResolverSettings,
    add_resolver_class,
)


class MockResolverSettings(ResolverSettings):
    """Settings type for MockResolver."""

    thresholds: dict[
        ModelStepName,
        Annotated[float, Field(ge=0.0, le=1.0)],
    ] = Field(default_factory=dict)

    def validate_inputs(self, model_names: Iterable[ModelStepName]) -> None:
        """Validate all model names are present in thresholds."""
        if missing := set(model_names) - set(self.thresholds.keys()):
            raise RuntimeError(f"Missing thresholds for models: {missing}")


def _connected_components_from_edges(
    model_edges: Mapping[ModelStepName, pl.DataFrame],
    thresholds: Mapping[ModelStepName, float],
) -> pl.DataFrame:
    """Generate clusters from model edge tables and per-model thresholds."""
    djs = DisjointSet[int]()

    for model_name, edges in model_edges.items():
        if edges.height == 0:
            continue

        threshold = thresholds[model_name]
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

    def compute_clusters(
        self, model_edges: Mapping[ModelStepName, pl.DataFrame]
    ) -> pl.DataFrame:
        """Compute mock clusters with connected components."""
        self.settings.validate_inputs(model_edges.keys())
        return _connected_components_from_edges(
            model_edges=model_edges,
            thresholds=self.settings.thresholds,
        )


add_resolver_class(MockResolver)


class ResolverTestkit(BaseModel):
    """Resolver plus local expected data for tests."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    resolver: Resolve
    assignments: pl.DataFrame
    entities: tuple[ClusterEntity, ...]

    @property
    def name(self) -> str:
        """Return resolver name."""
        return self.resolver.name

    def clean(self, *args: Any, **kwargs: Any) -> Clean:
        """Thin wrapper to build a cleaned view through this testkit's Resolve."""
        return self.resolver.clean(*args, **kwargs)


def resolver_factory(
    inputs: Iterable[ModelTestkit] | None = None,
    true_entities: Iterable[SourceEntity] | None = None,
    name: ResolverStepName | None = None,
    description: str | None = None,
    thresholds: Mapping[ModelStepName, float] | None = None,
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
        name: Name of the resolver. Defaults to a randomly generated word suffixed
            with '_resolver'.
        description: Description of the resolver.
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
        source_names: set[SourceStepName] = set(linked.sources.keys())
    else:
        inputs = tuple(inputs)
        source_names: set[SourceStepName] = set()
        for testkit in inputs:
            if not isinstance(testkit, ModelTestkit):
                raise TypeError("resolver_factory inputs must be ModelTestkit.")
            source_names.update(
                source.name
                for source in testkit.model.lineage()
                if type(source).__name__ == "Source"
            )

    input_map: dict[str, ModelTestkit] = {}
    for testkit in inputs:
        input_map.setdefault(testkit.name, testkit)

    resolver_inputs: list[Model] = []
    model_edges: dict[ModelStepName, pl.DataFrame] = {}
    for testkit in input_map.values():
        resolver_inputs.append(testkit.model)
        model_edges[testkit.name] = testkit.scores

    expected_model_names = set(input_map.keys())
    if thresholds is None:
        thresholds = {name: 0.0 for name in expected_model_names}

    if set(thresholds) != expected_model_names:
        raise ValueError("Threshold keys must exactly match resolver input models.")

    resolver_settings = MockResolverSettings(
        thresholds=thresholds,
    )

    generator = Faker()
    generator.seed_instance(seed)
    resolver = Resolve(
        *resolver_inputs,
        name=name or f"{generator.unique.word()}_resolver",
        resolver_class=MockResolver,
        resolver_settings=resolver_settings,
        description=description,
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
        assignments=assignments,
        entities=entities,
    )
