"""Resolve — collapse model edges into clusters and materialise the resolution."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

from matchlab.adapters import Adapter, Fingerprint
from matchlab.cleaning import Clean
from matchlab.core.exceptions import StepNotFound
from matchlab.core.logging import logger, profile_time
from matchlab.core.resolution import materialise_resolution
from matchlab.models import Model
from matchlab.resolvers.base import ResolverMethod, ResolverSettings
from matchlab.resolvers.components import Components
from matchlab.results import ResolverMatches
from matchlab.sources import Source
from matchlab.steps import Step

if TYPE_CHECKING:
    pass

_RESOLVER_CLASSES: dict[str, type[ResolverMethod]] = {}


def add_resolver_class(resolver_class: type[ResolverMethod]) -> None:
    """Register a resolver methodology so it can be named in a plan."""
    if not issubclass(resolver_class, ResolverMethod):
        raise ValueError("The argument is not a subclass of ResolverMethod.")
    _RESOLVER_CLASSES[resolver_class.__name__] = resolver_class


add_resolver_class(Components)


class Resolve(Step):
    """Clusters computed over one or more models' edges."""

    kind: ClassVar[str] = "resolver"

    def __init__(
        self,
        *models: Model,
        resolver_class: type[ResolverMethod] | str = "Components",
        resolver_settings: ResolverSettings | dict[str, Any] | None = None,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Define a resolver.

        Args:
            *models: The models whose edges to resolve. At least one.
            resolver_class: A `ResolverMethod` subclass or its registered name.
                Defaults to connected components.
            resolver_settings: Settings for that methodology.
            name: Optional plan name; derived from the first input when omitted.
            description: Optional human description.
        """
        deduped: list[Model] = []
        for model in models:
            if not isinstance(model, Model):
                raise ValueError(f"Resolver inputs must be models, got {type(model)}")
            if not any(model is seen for seen in deduped):
                deduped.append(model)
        if not deduped:
            raise ValueError("A resolver needs at least one model")

        self.inputs = tuple(deduped)
        self.description = description

        self.resolver_class = (
            _RESOLVER_CLASSES[resolver_class]
            if isinstance(resolver_class, str)
            else resolver_class
        )
        settings = resolver_settings if resolver_settings is not None else {}
        self.resolver_instance = self.resolver_class(settings=settings)
        if isinstance(settings, dict):
            settings_class = self.resolver_instance.__annotations__["settings"]
            self.resolver_settings = settings_class(**settings)
        else:
            self.resolver_settings = settings

        super().__init__(
            name=name or f"resolve_{self.inputs[0].name}", upstream=self.inputs
        )

    # -- Step contract ----------------------------------------------------------------

    def _config_key(self) -> bytes:
        return json.dumps(
            {
                "resolver_class": self.resolver_class.__name__,
                "resolver_settings": self.resolver_settings.model_dump(mode="json"),
                "inputs": [model.name for model in self.inputs],
            },
            sort_keys=True,
        ).encode()

    @profile_time(attr="name")
    def _execute(self, adapter: Adapter, fp: Fingerprint) -> None:
        logger.info("Computing clusters", prefix=f"Run {self.name}")
        edges = {model.name: adapter.read_model(model._fp) for model in self.inputs}
        clusters = self.resolver_instance.compute_clusters(model_edges=edges)

        # Every leaf reachable through this resolver's inputs — including records no
        # model formed an edge over. materialise_resolution carries those forward
        # (the merge-forward / fall-through requirement).
        upstream = pl.concat(
            [
                view.identifiers(adapter)
                for model in self.inputs
                for view in model.inputs
            ],
            how="vertical",
        ).unique()

        adapter.store_resolver(fp, materialise_resolution(clusters, upstream))

    # -- data -------------------------------------------------------------------------

    @property
    def sources(self) -> tuple[Source, ...]:
        """The sources reachable through this resolver, in lineage order."""
        return tuple(step for step in self.lineage() if isinstance(step, Source))

    def resolution(self) -> pl.DataFrame:
        """Return `(root, leaf, key, source)`. Collects the plan first if needed."""
        if not self.is_collected:
            self.collect()
        return self._require_adapter().read_resolver(self._fp)

    def results_eval(self) -> pl.DataFrame:
        """Return `(root, leaf)` for evaluation."""
        return self.resolution().select("root", "leaf").unique()

    def get_matches(
        self,
        source_filter: list[str] | None = None,
        location_names: list[str] | None = None,
    ) -> ResolverMatches:
        """Return the matches this resolver produced, optionally filtered."""
        resolution = self.resolution()
        available = {source.name: source for source in self.sources}

        names = list(available)
        if source_filter:
            names = [name for name in names if name in source_filter]
        if location_names:
            names = [
                name
                for name in names
                if available[name].location.config.name in location_names
            ]
        if not names:
            raise StepNotFound("No compatible source was found")

        return ResolverMatches(
            sources=[available[name] for name in names],
            query_results=[
                resolution.filter(pl.col("source") == name).select(
                    pl.col("root").alias("id"),
                    pl.col("key"),
                    pl.col("leaf").alias("leaf_id"),
                )
                for name in names
            ],
        )

    def lookup_key(
        self, from_source: str, to_sources: list[str], key: str
    ) -> dict[str, list[str]]:
        """Find the keys in `to_sources` that resolve to the same entity as `key`.

        Args:
            from_source: The source `key` belongs to.
            to_sources: The sources to find matching keys in.
            key: The key to look up.

        Returns:
            Source name → matching keys, including `from_source` itself.
        """
        resolution = self.resolution()
        origin = resolution.filter(
            (pl.col("source") == from_source) & (pl.col("key") == key)
        )
        if origin.height == 0:
            raise StepNotFound(f"Key '{key}' not found in source '{from_source}'")

        cluster = resolution.filter(pl.col("root") == origin["root"][0])

        def keys_in(source_name: str) -> list[str]:
            return cluster.filter(pl.col("source") == source_name)["key"].to_list()

        return {from_source: keys_in(from_source)} | {
            target: keys_in(target) for target in to_sources
        }

    # -- verbs ------------------------------------------------------------------------

    def clean(
        self,
        *sources: Source,
        cleaning: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> Clean:
        """Return a cleaned view of sources resolved *through* this resolver."""
        return Clean(
            *(sources or self.sources), resolver=self, cleaning=cleaning, **kwargs
        )
