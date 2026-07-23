"""Resolver nodes and methodology registry for client-side execution."""

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl
import pyarrow as pa

from matchbox.client.models.models import Model
from matchbox.client.queries import Query
from matchbox.client.resolvers.base import ResolverMethod, ResolverSettings
from matchbox.client.resolvers.components import Components
from matchbox.client.steps import StepABC, post_run
from matchbox.common.arrow import SCHEMA_CLUSTERS, check_schema_subset
from matchbox.common.dtos import (
    ResolverConfig,
    ResolverStepName,
    ResolverStepPath,
    SourceStepName,
    Step,
    StepName,
    StepType,
)
from matchbox.common.exceptions import MatchboxStepTypeError
from matchbox.common.hash import hash_clusters
from matchbox.common.logging import logger, profile_time
from matchbox.common.resolution import materialise_resolution

if TYPE_CHECKING:
    from matchbox.adapters import Adapter
    from matchbox.client.dags import DAG
    from matchbox.client.sources import Source
else:
    Adapter = Any
    DAG = Any
    Source = Any

_RESOLVER_CLASSES: dict[str, type[ResolverMethod]] = {}


def add_resolver_class(resolver_class: type[ResolverMethod]) -> None:
    """Register a resolver methodology class."""
    if not issubclass(resolver_class, ResolverMethod):
        raise ValueError("The argument is not a proper subclass of ResolverMethod.")
    _RESOLVER_CLASSES[resolver_class.__name__] = resolver_class
    logger.debug(f"Registered resolver class: {resolver_class.__name__}")


add_resolver_class(Components)


class Resolver(StepABC):
    """Client-side node that computes clusters from model and resolver inputs."""

    _local_data_schema: ClassVar[pa.Schema] = SCHEMA_CLUSTERS
    _kind_tag: ClassVar[str] = "resolver"

    def __init__(
        self,
        dag: DAG,
        name: ResolverStepName,
        inputs: Iterable[Model],
        resolver_class: type[ResolverMethod] | str,
        resolver_settings: ResolverSettings | dict[str, Any],
        description: str | None = None,
    ) -> None:
        """Create a resolver node that computes clusters from its inputs."""
        super().__init__(dag=dag, name=ResolverStepName(name), description=description)

        deduped_inputs: list[Model] = []
        seen_names: set[str] = set()
        for node in inputs:
            if not isinstance(node, Model):
                raise MatchboxStepTypeError(
                    step_name=getattr(node, "name", node),
                    expected_step_types=[StepType.MODEL],
                )
            if node.name in seen_names:
                continue
            seen_names.add(node.name)
            deduped_inputs.append(node)
        self.inputs = tuple(deduped_inputs)

        if len(self.inputs) < 1:
            raise ValueError("Resolver needs at least one input")

        if isinstance(resolver_class, str):
            self.resolver_class: type[ResolverMethod] = _RESOLVER_CLASSES[
                resolver_class
            ]
        else:
            self.resolver_class = resolver_class

        self.resolver_instance = self.resolver_class(settings=resolver_settings)

        if isinstance(resolver_settings, dict):
            SettingsClass = self.resolver_instance.__annotations__["settings"]
            self.resolver_settings = SettingsClass(**resolver_settings)
        else:
            self.resolver_settings = resolver_settings

        # The complete, merge-forward resolution (root, leaf, key, source), computed on
        # run() and persisted on sync(). This is what get_matches / lookup_key read.
        self._resolution: pl.DataFrame | None = None

    @property
    def results(self) -> pl.DataFrame | None:
        """The locally computed cluster assignments. Alias for local_data."""
        return self._local_data

    @results.setter
    def results(self, value: pl.DataFrame | None) -> None:
        self._local_data = value

    @property
    def results_eval(self) -> pl.DataFrame:
        """Mapping of resolved root clusters to source leaf IDs, for evaluation."""
        if self._resolution is None:
            raise RuntimeError(
                "The resolver must be run before requesting evaluation results."
            )
        return self._resolution.select("root", "leaf").unique()

    @property
    def config(self) -> ResolverConfig:
        """Generate config DTO from Resolver."""
        return ResolverConfig(
            resolver_class=self.resolver_class.__name__,
            resolver_settings=self.resolver_settings.model_dump(mode="json"),
            inputs=tuple(node.name for node in self.inputs),
        )

    @property
    def sources(self) -> set[SourceStepName]:
        """Set of source names upstream of this node."""
        upstream: set[SourceStepName] = set()
        for node in self.inputs:
            upstream.update(node.sources)
        return upstream

    @property
    def path(self) -> ResolverStepPath:
        """Return resolver step path."""
        return ResolverStepPath(
            collection=self.dag.name,
            run=self.dag.run,
            name=self.name,
        )

    @profile_time(attr="name")
    def compute_clusters(
        self, model_edges: Mapping[StepName, pl.DataFrame]
    ) -> pl.DataFrame:
        """Delegate cluster computation to the configured resolver instance."""
        return self.resolver_instance.compute_clusters(model_edges=model_edges)

    @profile_time(attr="name")
    def run(self) -> pl.DataFrame:
        """Run the resolver, computing clusters and the complete resolution.

        Computes connected-component clusters over the input models' edges, then
        materialises the complete, merge-forward resolution `(root, leaf, key, source)`
        from those clusters and the upstream resolution of the inputs' queries. Leaves
        grouped upstream but untouched here are preserved (fall-through).
        """
        model_edges: dict[StepName, pl.DataFrame] = {}

        for node in self.inputs:
            if node.results is None:
                raise ValueError(
                    f"Resolver input '{node.name}' has no local results. "
                    "Run upstream nodes before running this resolver."
                )
            model_edges[node.name] = node.results

        self._local_data = self.compute_clusters(model_edges=model_edges)
        self._resolution = materialise_resolution(
            clusters=self._local_data, upstream=self._gather_upstream()
        )
        return self._local_data

    def _gather_upstream(self) -> pl.DataFrame:
        """Union the input models' cached query resolutions into one upstream table.

        Covers every leaf reachable by this resolver's inputs — including records no
        model formed an edge over — so the resolution can carry them forward.
        """
        frames: list[pl.DataFrame] = []
        for node in self.inputs:
            for query in (node.left_query, node.right_query):
                if query is None:
                    continue
                if query._upstream is None:
                    raise RuntimeError(
                        f"Model '{node.name}' has no cached query data. "
                        "Resolvers require inputs run with cache_leaf_ids enabled "
                        "(the default; disabled by low_memory)."
                    )
                frames.append(query._upstream)
        return pl.concat(frames, how="vertical").unique()

    def _store(self, adapter: Adapter) -> None:
        """Persist the complete resolution to the adapter."""
        adapter.store_resolver(self._fp, self._resolution)

    @post_run
    def _fingerprint(self) -> bytes:
        """Compute resolver fingerprint from semantic cluster membership."""
        check_schema_subset(
            expected=self._local_data_schema, actual=self._local_data.to_arrow().schema
        )
        return hash_clusters(self._local_data.to_arrow())

    @post_run
    def to_dto(self) -> Step:
        """Convert to Step DTO for API calls."""
        return Step(
            description=self.description,
            step_type=StepType.RESOLVER,
            config=self.config,
            fingerprint=self._fingerprint(),
        )

    @classmethod
    def from_dto(
        cls,
        step: Step,
        step_name: str,
        dag: DAG,
        **kwargs: Any,
    ) -> "Resolver":
        """Reconstruct from Step DTO."""
        if step.step_type != StepType.RESOLVER:
            raise ValueError("Step must be of type 'resolver'")

        return cls(
            dag=dag,
            name=ResolverStepName(step_name),
            description=step.description,
            inputs=[dag.nodes[name] for name in step.config.inputs],
            resolver_class=step.config.resolver_class,
            resolver_settings=step.config.resolver_settings,
        )

    def query(self, *sources: Source, **kwargs: Any) -> Query:
        """Create a query rooted at this resolver."""
        if not sources:
            sources = tuple(self.dag.get_source(name) for name in sorted(self.sources))
        return Query(*sources, resolver=self, dag=self.dag, **kwargs)
