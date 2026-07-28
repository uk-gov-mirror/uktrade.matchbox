"""Resolve — collapse model edges into clusters and materialise the resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Self

import polars as pl

from matchlab.adapters import Adapter, Fingerprint
from matchlab.core.exceptions import StepNotFound
from matchlab.core.resolution import materialise_resolution
from matchlab.models import Model
from matchlab.resolvers.base import ResolverMethod, ResolverSettings
from matchlab.resolvers.components import Components
from matchlab.results import ResolverMatches
from matchlab.sources import Source
from matchlab.specs import ResolverSpec
from matchlab.steps import Step
from matchlab.views import View

if TYPE_CHECKING:
    pass

_RESOLVER_CLASSES: dict[str, type[ResolverMethod]] = {}


def add_resolver_class(resolver_class: type[ResolverMethod]) -> None:
    """Register a resolver methodology so it can be named in a plan."""
    if not issubclass(resolver_class, ResolverMethod):
        raise ValueError("The argument is not a subclass of ResolverMethod.")
    _RESOLVER_CLASSES[resolver_class.__name__] = resolver_class


add_resolver_class(Components)


class Resolver(Step):
    """Clusters computed over one or more models' edges."""

    kind: ClassVar[str] = "resolver"

    def __init__(
        self,
        *models: Model,
        resolver_class: type[ResolverMethod] | str = "Components",
        resolver_settings: ResolverSettings | dict[str, Any] | None = None,
    ) -> None:
        """Define a resolver.

        Args:
            *models: The models whose edges to resolve. At least one.
            resolver_class: A `ResolverMethod` subclass or its registered name.
                Defaults to connected components.
            resolver_settings: Settings for that methodology.
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

        self.resolver_class = (
            _RESOLVER_CLASSES[resolver_class]
            if isinstance(resolver_class, str)
            else resolver_class
        )
        settings = resolver_settings if resolver_settings is not None else {}
        if isinstance(settings, dict):
            settings_class = self.resolver_class.__annotations__["settings"]
            self.resolver_settings = settings_class(
                **{
                    field: self._positions(field, value)
                    for field, value in settings.items()
                }
            )
        else:
            self.resolver_settings = settings
        self.resolver_instance = self.resolver_class(settings=self.resolver_settings)

        super().__init__(upstream=self.inputs)

    def _positions(self, field: str, value: Any) -> Any:  # noqa: ANN401 - any setting
        """Replace `Model` keys in a setting with the position of that input.

        A setting that points at one of this resolver's inputs — per-model thresholds,
        say — is written as `{model: 0.9}`, holding the object you already have rather
        than retyping its name. Here is the only place that can be translated, because
        here is where both the models and the order they were given in are known.

        Positions rather than names because a name is not identity: renaming a model
        would otherwise move this resolver's fingerprint without changing a byte of its
        output. Reordering the inputs *does* reassign thresholds, but reordering already
        changes the fingerprint — parents are folded in order — so that is a different
        resolver, not one behaving inconsistently.

        Raises:
            ValueError: If a setting names a model that is not an input.
        """
        if not isinstance(value, dict):
            return value

        position = {id(model): index for index, model in enumerate(self.inputs)}
        translated: dict[Any, Any] = {}
        for key, setting in value.items():
            if not isinstance(key, Model):
                translated[key] = setting
                continue
            if id(key) not in position:
                raise ValueError(
                    f"'{field}' has an entry for a model this resolver does not read. "
                    f"It resolves {len(self.inputs)} model(s), and a setting may only "
                    "point at one of those."
                )
            translated[position[id(key)]] = setting
        return translated

    # -- Step contract ----------------------------------------------------------------

    def __str__(self) -> str:
        """A resolver is drawn with the methodology implementing it."""
        return f"{self.kind}({self.resolver_class.__name__})"

    @property
    def spec(self) -> ResolverSpec:
        """The serialisable spec for this resolver."""
        return ResolverSpec(
            resolver_class=self.resolver_class.__name__,
            resolver_settings=self.resolver_settings.model_dump(mode="json"),
        )

    # -- publishing -------------------------------------------------------------------

    def publish(self, label: str, overwrite: bool = False) -> Self:
        """Point a label at this resolution, so it can be found without the plan.

        Publishing is an act, not a property of the plan: a label changes nothing about
        what gets computed, and there is nothing to point at until the resolution
        exists. So it happens after collection — `resolver.collect().publish("x")` —
        and a plan that is never published is still perfectly runnable, just unlabelled.

        A *label* rather than a name, because a name is something else here: a source's
        name is part of its output, prefixing every column it contributes. A label
        belongs to the store, and points at whichever resolution you last aimed it at.

        Re-publishing the same label for the same resolution is a no-op, so re-running
        an unchanged pipeline is safe. Aiming an existing label at a *different*
        resolution needs `overwrite=True`, because that is how you lose track of what a
        label used to mean.

        Args:
            label: The label to publish under.
            overwrite: Move the label if it already points somewhere else.

        Returns:
            This resolver, so it chains off `collect()`.

        Raises:
            RuntimeError: If this resolver has not been collected.
            ValueError: If `label` already points at a different resolution and
                `overwrite` is not set.
        """
        adapter, fp = self._collected()
        existing = adapter.find(label)
        if existing is not None and existing != fp and not overwrite:
            raise ValueError(
                f"The label '{label}' already points at a different resolution "
                f"({existing.hex()[:8]}). Pass overwrite=True to move it, or publish "
                "under another label."
            )
        adapter.publish(label, fp)
        return self

    # -- Step contract ----------------------------------------------------------------

    def _execute(self, adapter: Adapter, fp: Fingerprint) -> None:
        edges = {
            position: adapter.read_model(model._fp)
            for position, model in enumerate(self.inputs)
        }
        clusters = self.resolver_instance.compute_clusters(model_edges=edges)

        # Every leaf reachable through this resolver's inputs — including records no
        # model formed an edge over. materialise_resolution carries those forward
        # (the merge-forward / fall-through requirement).

        # Deduplicate what is read. Linking every pair of n sources gives n(n-1)
        # (model, view) pairs but only a handful of distinct readings between them:
        # they share an upstream resolver and cover the same sources, so asking per pair
        # repeats the same query a quadratic number of times. `dict.fromkeys` dedupes
        # while keeping lineage order, so the frame is built the same way every run.
        reads = dict.fromkeys(
            read
            for model in self.inputs
            for view in model.inputs
            for read in view._identifier_reads
        )
        upstream = pl.concat(
            [adapter.read_identifiers(*read) for read in reads],
            how="vertical",
        ).unique()

        # Record which source artifacts this resolution covers. A resolution names
        # its sources, but a store can hold several generations of a name — this is
        # what lets it be read back without the plan that built it. Those names are
        # data, tagging which source each row came from; they are nothing to do with
        # publishing, which is `publish()` and happens after this.
        adapter.store_resolver(
            fp=fp,
            resolution=materialise_resolution(clusters, upstream),
            sources={source.name: source._fp for source in self.sources},
        )

    # -- data -------------------------------------------------------------------------

    @property
    def sources(self) -> tuple[Source, ...]:
        """The sources reachable through this resolver, in lineage order."""
        return tuple(step for step in self.lineage() if isinstance(step, Source))

    def resolution(self) -> pl.DataFrame:
        """Return `(root, leaf, key, source)`. Collects the plan first if needed."""
        if not self.is_collected:
            self.collect()
        adapter, fp = self._collected()
        return adapter.read_resolver(fp)

    def results_eval(self) -> pl.DataFrame:
        """Return `(root, leaf)` for evaluation."""
        return self.resolution().select("root", "leaf").unique()

    def get_matches(self, source_filter: list[str] | None = None) -> ResolverMatches:
        """Return the matches this resolver produced, optionally filtered."""
        resolution = self.resolution()
        available = {source.name: source for source in self.sources}

        names = list(available)
        if source_filter:
            names = [name for name in names if name in source_filter]
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

    def view(
        self,
        *sources: Source,
        cleaning: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> View:
        """Return a cleaned view of sources resolved *through* this resolver."""
        return View(
            *(sources or self.sources), resolver=self, cleaning=cleaning, **kwargs
        )
