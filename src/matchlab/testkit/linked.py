"""Sources that share entities, and everything you do with them.

This is the entry point. `linked_sources_factory` generates several sources from one
pool of planted entities, so the same real-world thing appears in more than one of them
under different keys. That is what makes linking testable at all.

The object it returns is the only one that knows the answer, so everything needing the
answer hangs off it. Build a plan with `dedupe()`/`link()` and score one with
`diff_resolver_output()`/`diff_model_edges()`, none of which need to be told the truth.
"""

import warnings
from collections.abc import Callable, Iterable
from functools import cache
from typing import Self

import polars as pl
from faker import Faker
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Engine, create_engine
from sqlglot import cast, select
from sqlglot.expressions import column

from matchlab.models.models import Model
from matchlab.recordstep import RecordStep
from matchlab.resolvers import Resolver
from matchlab.resources import Resource
from matchlab.sources import read_database
from matchlab.testkit._generate import generate_entities, generate_source
from matchlab.testkit.compare import (
    diff_entities,
    resolver_output_to_clusters,
    scores_to_clusters,
)
from matchlab.testkit.entities import Cluster, EntityReference, TrueEntity
from matchlab.testkit.features import FeatureConfig, SourceParameters, SuffixRule
from matchlab.testkit.matchers import (
    AnswerKey,
    PerfectDeduper,
    PerfectLinker,
)
from matchlab.testkit.models import GeneratedModel, generate_entity_scores
from matchlab.testkit.sources import GeneratedSource


class LinkedSources(BaseModel):
    """A set of generated sources, plus the true entities planted across all of them.

    This is the object the module docstring describes. Build a plan from it with
    `dedupe()`/`link()`, then score one with
    `diff_resolver_output()`/`diff_model_edges()`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    true_entities: set[TrueEntity] = Field(default_factory=set)
    sources: dict[str, GeneratedSource]

    def find_entities(
        self,
        min_appearances: dict[str, int] | None = None,
        max_appearances: dict[str, int] | None = None,
    ) -> list[TrueEntity]:
        """Find entities matching appearance criteria.

        Args:
            min_appearances: For each named source, the minimum number of keys an
                entity must have there.
            max_appearances: For each named source, the maximum number of keys an
                entity may have there.
        """
        result = list(self.true_entities)

        def _meets_criteria(
            entity: TrueEntity, criteria: dict[str, int], compare: Callable
        ) -> bool:
            return all(
                compare(len(entity.get_keys(src)), count)
                for src, count in criteria.items()
            )

        if min_appearances:
            result = [
                entity
                for entity in result
                if _meets_criteria(entity, min_appearances, lambda x, y: x >= y)
            ]

        if max_appearances:
            result = [
                entity
                for entity in result
                if _meets_criteria(entity, max_appearances, lambda x, y: x <= y)
            ]

        return result

    def true_entity_subset(self, *sources: str) -> list[Cluster]:
        """Return a subset of true entities that appear in the given sources."""
        cluster_entities = [entity.cluster(*sources) for entity in self.true_entities]
        return [entity for entity in cluster_entities if entity is not None]

    def diff_model_edges(
        self,
        edges: pl.DataFrame,
        left: GeneratedSource,
        right: GeneratedSource | None = None,
        threshold: float = 0.0,
    ) -> tuple[bool, dict]:
        """Diff a model's edges against the planted true entities.

        This takes the sources rather than their innards. The clusters the model
        started from, and the names to compare over, are both properties of the
        sources you handed it, so re-supplying them separately was only a chance to
        pass the wrong ones.

        Args:
            edges: The model edge table to score.
            left: The generated source the model read as its left input.
            right: Its right input, for a linker. `None` for a deduper.
            threshold: Score at or above which an edge counts as a match.

        Returns:
            `(identical, report)`. See
                [`diff_entities()`][matchlab.testkit.compare.diff_entities]
                for the report format.
        """
        names = [left.source.name] + ([right.source.name] if right is not None else [])
        return diff_entities(
            expected=self.true_entity_subset(*names),
            actual=scores_to_clusters(
                scores=edges,
                left_clusters=left.input_clusters,
                right_clusters=right.input_clusters if right is not None else None,
                threshold=threshold,
            ),
        )

    def diff_resolver_output(
        self, resolver_output: pl.DataFrame, *sources: str
    ) -> tuple[bool, dict]:
        """Diff a collected resolver output against the planted true entities.

        The counterpart to `diff_model_edges` for a plan that has run. It scores the
        `(root, key, source)` table `Resolver.entities()` returns, whose IDs are
        content-derived at collect time and so have no counterpart here. Only the
        record membership is comparable, which is exactly what a cluster asserts.

        Args:
            resolver_output: The table returned by `Resolver.entities()`.
            *sources: The source names to compare over, e.g. `"crn", "cdms"`.

        Returns:
            `(identical, report)`. See
                [`diff_entities()`][matchlab.testkit.compare.diff_entities]
                for the report format.
        """
        return diff_entities(
            expected=self.true_entity_subset(*sources),
            actual=list(resolver_output_to_clusters(resolver_output)),
        )

    def write_to_location(self) -> Self:
        """Write every source's data to its location.

        Mirrors `GeneratedSource.write_to_location`, so callers don't need to loop over
        `sources` by hand.
        """
        for source_testkit in self.sources.values():
            source_testkit.write_to_location()

        return self

    def dedupe(
        self,
        source: str,
        *,
        true_entities: Iterable[TrueEntity] | None = None,
        score_range: tuple[float, float] = (0.8, 1.0),
        seed: int = 42,
    ) -> "GeneratedModel":
        """Build a perfect deduper over one of these sources.

        The truth is implied. It is what this testkit planted, so there is nothing to
        thread through by hand.

        Args:
            source: Name of the source to deduplicate.
            true_entities: Restrict the truth to these entities. Defaults to all of
                them, which is what you want unless you are deliberately building a
                model that knows only part of the answer.
            score_range: Range the emitted scores fall in.
            seed: Random seed for the generated scores.
        """
        left = self.sources[source]
        return _build_model(
            left_testkit=left,
            right_testkit=None,
            true_entities=(
                self.true_entities if true_entities is None else true_entities
            ),
            left=left.source,
            right=None,
            score_range=score_range,
            seed=seed,
        )

    def link(
        self,
        left: str,
        right: str,
        *,
        through: "Resolver | None" = None,
        true_entities: Iterable[TrueEntity] | None = None,
        score_range: tuple[float, float] = (0.8, 1.0),
        seed: int = 42,
    ) -> "GeneratedModel":
        """Build a perfect linker between two of these sources.

        Args:
            left: Name of the left source.
            right: Name of the right source.
            through: Read the *left* source through this resolver rather than raw, so
                the link sits on top of an upstream dedupe. That is the layered shape
                worth exercising: the apex must carry the upstream grouping forward as
                well as its own.
            true_entities: Restrict the truth to these entities. Defaults to all.
            score_range: Range the emitted scores fall in.
            seed: Random seed for the generated scores.
        """
        left_testkit, right_testkit = self.sources[left], self.sources[right]
        return _build_model(
            left_testkit=left_testkit,
            right_testkit=right_testkit,
            true_entities=(
                self.true_entities if true_entities is None else true_entities
            ),
            left=(through if through is not None else left_testkit.source),
            right=right_testkit.source,
            score_range=score_range,
            seed=seed,
        )


def _build_model(
    *,
    left_testkit: GeneratedSource,
    right_testkit: GeneratedSource | None,
    true_entities: Iterable[TrueEntity],
    left: RecordStep,
    right: RecordStep | None,
    score_range: tuple[float, float] = (0.8, 1.0),
    seed: int = 42,
) -> GeneratedModel:
    """Wire a scripted model onto generated sources, building both expectations.

    You usually want `LinkedSources.dedupe` or `.link` instead. They know the truth and
    the record steps, so they take a source name rather than seven arguments. This is
    the shared implementation behind them, and is private because building the record
    steps correctly is the part that needs the testkit.

    Args:
        left_testkit: Generated testkit for the left source. Value-keyed truth is
            derived from it, so it cannot be a bare `Source`.
        right_testkit: The right source's testkit, for a linker. `None` deduplicates.
        true_entities: The planted entities the model should recover.
        left: The record step the model reads as its left input.
        right: The record step the model reads as its right input, for a linker.
        score_range: Range the emitted scores fall in.
        seed: Random seed for the generated scores.

    Raises:
        ValueError: If `score_range` is not increasing within [0, 1].
    """
    if not (0 <= score_range[0] <= score_range[1] <= 1):
        raise ValueError("Scores must be increasing values between 0 and 1")

    entities = tuple(true_entities)
    left_entities = left_testkit.input_clusters
    right_entities = right_testkit.input_clusters if right_testkit is not None else None

    # The synthetic-ID expectation, in the testkit's own entity space. This is what
    # `diff_model_edges` asserts a methodology against without running a plan.
    scores = generate_entity_scores(
        left_entities=frozenset(left_entities),
        right_entities=frozenset(right_entities) if right_entities else None,
        true_entities=frozenset(entities),
        score_range=score_range,
        seed=seed,
    )

    # The value-keyed truth. The model matches on row values, so a collected plan
    # emits the same truth over whatever content-derived IDs the record step actually
    # carries.
    truth_id = AnswerKey.from_sources(
        left=left_testkit,
        true_entities=entities,
        right=right_testkit,
        score=score_range[1],
    ).register()

    if right_testkit is not None:
        model_class: type[PerfectLinker] | type[PerfectDeduper] = PerfectLinker
    else:
        model_class = PerfectDeduper

    return GeneratedModel(
        model=Model(
            model_class=model_class,
            model_settings={"truth_id": truth_id},
            left=left,
            right=right,
        ),
        left_source=left_testkit,
        right_source=right_testkit,
        scores=scores,
    )


@cache
def linked_sources_factory(
    source_parameters: tuple[SourceParameters, ...] | None = None,
    n_true_entities: int | None = None,
    engine: Engine | None = None,
    seed: int = 42,
) -> LinkedSources:
    """Generate a set of linked sources with tracked entities.

    Args:
        source_parameters: Optional tuple of source testkit parameters
        n_true_entities: Optional number of true entities to generate. If provided,
            overrides any n_true_entities in source configs. If not provided, each
            SourceParameters must specify its own n_true_entities.
        engine: Optional SQLAlchemy engine to use for all sources. If provided,
            overrides any engine in source configs.
        seed: Random seed for reproducibility
    """
    generator = Faker()
    generator.seed_instance(seed)

    default_engine = create_engine("sqlite:///:memory:")

    if source_parameters is None:
        # Use factory parameter or default for default configs
        n_true_entities = n_true_entities if n_true_entities is not None else 10

        features = {
            "company_name": FeatureConfig(
                name="company_name",
                base_generator="company",
            ),
            "crn": FeatureConfig(
                name="crn",
                base_generator="bothify",
                parameters=(("text", "???-###-???-###"),),
            ),
            "dh": FeatureConfig(
                name="dh",
                base_generator="numerify",
                parameters=(("text", "########"),),
            ),
            "cdms": FeatureConfig(
                name="cdms",
                base_generator="numerify",
                parameters=(("text", "ORG-########"),),
            ),
            "address": FeatureConfig(
                name="address",
                base_generator="address",
            ),
        }

        source_parameters = (
            SourceParameters(
                name="crn",
                engine=engine or default_engine,
                features=(
                    features["company_name"].add_variations(
                        SuffixRule(suffix=" Limited"),
                        SuffixRule(suffix=" UK"),
                        SuffixRule(suffix=" Company"),
                    ),
                    features["crn"],
                ),
                n_true_entities=n_true_entities,
                repetition=0,
            ),
            SourceParameters(
                name="dh",
                engine=engine or default_engine,
                features=(
                    features["company_name"],
                    features["dh"],
                ),
                n_true_entities=n_true_entities // 2,
                repetition=0,
            ),
            SourceParameters(
                name="cdms",
                engine=engine or default_engine,
                features=(
                    features["crn"],
                    features["cdms"],
                ),
                n_true_entities=n_true_entities,
                repetition=1,
            ),
        )
    else:
        if n_true_entities is not None:
            # Factory parameter provided, warn if configs have values set
            config_entities = [
                parameters.n_true_entities for parameters in source_parameters
            ]
            if any(n is not None for n in config_entities):
                warnings.warn(
                    "Both source testkit configs and linked_sources_factory specify "
                    "n_true_entities. The factory parameter will be used.",
                    UserWarning,
                    stacklevel=2,
                )
            # Override all configs with factory parameter
            source_parameters = tuple(
                parameters.model_copy(
                    update={
                        "engine": engine or parameters.engine,
                        "n_true_entities": n_true_entities,
                    }
                )
                for parameters in source_parameters
            )
        else:
            # No factory parameter, check all configs have n_true_entities set
            missing_counts = [
                parameters.name
                for parameters in source_parameters
                if parameters.n_true_entities is None
            ]
            if missing_counts:
                raise ValueError(
                    "n_true_entities not set for sources: "
                    f"{', '.join(missing_counts)}. When factory n_true_entities is "
                    "not provided, all configs must specify it."
                )

    # Collect all unique features
    all_features = set()
    for parameters in source_parameters:
        all_features.update(parameters.features)
    all_features = tuple(sorted(all_features, key=lambda f: f.name))

    # Find maximum number of entities needed across all sources
    max_entities = max(parameters.n_true_entities for parameters in source_parameters)

    # Generate all possible entities
    all_entities = generate_entities(
        generator=generator, features=all_features, n=max_entities
    )

    # Initialize LinkedSources
    true_entity_lookup = {entity.id: entity for entity in all_entities}
    linked = LinkedSources(
        true_entities=all_entities,
        sources={},
    )

    # Generate sources
    for parameters in source_parameters:
        # Generate source data using seed entities
        data, entity_keys, row_keys = generate_source(
            generator=generator,
            features=tuple(parameters.features),
            n_true_entities=parameters.n_true_entities,
            repetition=parameters.repetition,
            seed_entities=all_entities,
        )

        # Create Cluster objects from row_keys
        clusters = [
            Cluster(
                id=row_id,
                keys=EntityReference({parameters.name: frozenset(keys)}),
            )
            for row_id, keys in row_keys.items()
        ]

        # Create source config. Resolve the engine here rather than relying on a field
        # default, so sources that were given none still share this call's warehouse
        # and can therefore be linked to each other.
        source = read_database(
            parameters.name,
            sql=select(
                cast(column("key"), "string").as_("key"),
                *[column(feature.name) for feature in parameters.features],
            )
            .from_(parameters.name)
            .sql(),
            client=Resource(
                str(parameters.name),
                parameters.engine or engine or default_engine,
            ),
            key_field="key",
        )

        # Add source to linked.sources
        linked.sources[parameters.name] = GeneratedSource(
            source=source,
            features=tuple(parameters.features),
            data=data,
            input_clusters=tuple(sorted(clusters)),
        )

        # Update entities with source references
        for entity_id, keys in entity_keys.items():
            entity = true_entity_lookup[entity_id]
            entity.add_source_reference(parameters.name, keys)

    return linked
