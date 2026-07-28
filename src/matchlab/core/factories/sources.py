"""Factories for generating sources and linked source testkits for testing."""

import warnings
from collections.abc import Callable, Mapping
from functools import cache, wraps
from itertools import product
from math import prod
from typing import Any, ParamSpec, Self, TypeVar

import pandas as pd
import polars as pl
import pyarrow as pa
from adbc_driver_manager.dbapi import Connection as AdbcConnection
from faker import Faker
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Engine, create_engine
from sqlglot import cast, select
from sqlglot.expressions import column

from matchlab.core.factories.entities import (
    ClusterEntity,
    EntityReference,
    FeatureConfig,
    SourceEntity,
    SuffixRule,
    diff_entities,
    generate_entities,
    scores_to_results_entities,
)
from matchlab.locations import RelationalDBLocation
from matchlab.sources import Source
from matchlab.specs import SourceSpec
from matchlab.views import View

P = ParamSpec("P")
R = TypeVar("R")


def make_features_hashable(func: Callable[P, R]) -> Callable[P, R]:
    """Decorator to allow configuring source_factory with dicts.

    This retains the hashability of FeatureConfig while still making it simple
    to use the factory without special objects.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        # Handle features in first positional arg
        if args and args[0] is not None:
            if isinstance(args[0][0], dict):
                args = (tuple(FeatureConfig(**d) for d in args[0]),) + args[1:]
            else:
                args = (tuple(args[0]),) + args[1:]

        # Handle features in kwargs
        if "features" in kwargs and kwargs["features"] is not None:
            if isinstance(kwargs["features"][0], dict):
                kwargs["features"] = tuple(
                    FeatureConfig(**d) for d in kwargs["features"]
                )
            else:
                kwargs["features"] = tuple(kwargs["features"])

        return func(*args, **kwargs)

    return wrapper


class SourceTestkitParameters(BaseModel):
    """Configuration for generating a source."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    features: tuple[FeatureConfig, ...] = Field(default_factory=tuple)
    name: str
    engine: Engine | AdbcConnection = Field(default=create_engine("sqlite:///:memory:"))
    n_true_entities: int | None = Field(default=None)
    repetition: int = Field(default=0)


class SourceTestkit(BaseModel):
    """A testkit of data and metadata for a `Source`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: Source = Field(
        description="The Source object containing the spec and convenience methods."
    )
    features: tuple[FeatureConfig, ...] | None = Field(
        description=(
            "The features used to generate the data. "
            "If None, the source data was not generated, but set manually."
        ),
        default=None,
    )
    data: pa.Table = Field(
        description="Data corresponding to the output of queries, without leaf data."
    )
    entities: tuple[ClusterEntity, ...] = Field(
        description="ClusterEntities that were generated from the source."
    )

    @property
    def name(self) -> str:
        """Return the source name."""
        return self.source.name

    @property
    def source_spec(self) -> SourceSpec:
        """Return the SourceSpec from the source."""
        return self.source.spec

    @property
    def field_names(self) -> list[str]:
        """The non-key columns of this testkit's data, in generation order.

        Taken from the testkit rather than from `Source.index_fields`, which would
        have to read the warehouse — these are needed while building a plan. Falls
        back to the data's own columns when the source was set manually rather than
        generated from features.
        """
        if self.features is not None:
            return [feature.name for feature in self.features]
        return [
            column for column in self.data.column_names if column not in ("key", "id")
        ]

    def view(self, *args: Any, **kwargs: Any) -> View:
        """Thin wrapper to build a cleaned view over this testkit's Source."""
        return self.source.view(*args, **kwargs)

    def write_to_location(self, client: Any | None = None) -> Self:  # noqa: ANN401
        """Write the data to the source's location.

        Args:
            client: Write through this client instead of the source's own, rebuilding
                the source's location around it. Testkits are generated against an
                in-memory SQLite engine by default, so a test that wants a real
                warehouse repoints one here.
        """
        if client:
            self.source.location = type(self.source.location)(
                name=self.source.location.name, client=client
            )

        pl.from_arrow(self.data).write_database(
            table_name=self.source.name,
            connection=self.source.location.client,
            if_table_exists="replace",
        )

        return self


class LinkedSourcesTestkit(BaseModel):
    """Container for multiple related source testkits with entity tracking."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    true_entities: set[SourceEntity] = Field(default_factory=set)
    sources: dict[str, SourceTestkit]

    def find_entities(
        self,
        min_appearances: dict[str, int] | None = None,
        max_appearances: dict[str, int] | None = None,
    ) -> list[SourceEntity]:
        """Find entities matching appearance criteria."""
        result = list(self.true_entities)

        def _meets_criteria(
            entity: SourceEntity, criteria: dict[str, int], compare: Callable
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

    def true_entity_subset(self, *sources: str) -> list[ClusterEntity]:
        """Return a subset of true entities that appear in the given sources."""
        cluster_entities = [
            entity.to_cluster_entity(*sources) for entity in self.true_entities
        ]
        return [entity for entity in cluster_entities if entity is not None]

    def diff_model_edges(
        self,
        scores: pl.DataFrame,
        sources: list[str],
        left_clusters: tuple[ClusterEntity, ...],
        right_clusters: tuple[ClusterEntity, ...] | None = None,
        threshold: float = 0.0,
    ) -> tuple[bool, dict]:
        """Diff model edge outputs with the true SourceEntities.

        Args:
            scores: Model edge table to diff
            sources: Subset of the LinkedSourcesTestkit.sources that represents
                the true sources to compare against
            left_clusters: ClusterEntity objects from the object used as an input
                to the process that produced the model edge table. Should
                be a SourceTestkit.entities or ResolverTestkit.entities
            right_clusters: ClusterEntity objects from the object used as an input
                to the process that produced the model edge table. Should
                be a SourceTestkit.entities or ResolverTestkit.entities
            threshold: Threshold for considering a match true

        Returns:
            A tuple of whether the results are identical, and a report dictionary. See
                [`diff_entities()`][matchlab.core.factories.entities.diff_entities]
                for the report format.
        """
        return diff_entities(
            expected=self.true_entity_subset(*sources),
            actual=scores_to_results_entities(
                scores=scores,
                left_clusters=left_clusters,
                right_clusters=right_clusters,
                threshold=threshold,
            ),
        )

    def diff_clusters(
        self,
        assignments: pl.DataFrame,
        sources: list[str],
        input_clusters: Mapping[str, tuple[ClusterEntity, ...]],
    ) -> tuple[bool, dict]:
        """Diff cluster assignments with the true SourceEntities.

        Args:
            assignments: Cluster assignment table to diff
            sources: Subset of the LinkedSourcesTestkit.sources that represents
                the true sources to compare against
            input_clusters: raw input ClusterEntity from the
                ModelTestkit.left/right_clusters used as inputs in to the resolver
                that produced the assignments table

        Returns:
            A tuple of whether the results are identical, and a report dictionary. See
                [`diff_entities()`][matchlab.core.factories.entities.diff_entities]
                for the report format.
        """
        id_to_entity: dict[int, ClusterEntity] = {
            entity.id: entity
            for entities in input_clusters.values()
            for entity in entities
        }
        actual = [
            sum(id_to_entity[child_id] for child_id in group["child_id"].to_list())
            for group in assignments.partition_by("parent_id")
        ]
        return diff_entities(
            expected=self.true_entity_subset(*sources),
            actual=actual,
        )

    def write_to_location(self) -> Self:
        """Write the data to the source's location."""
        for source_testkit in self.sources.values():
            source_testkit.write_to_location()

        return self


def generate_rows(
    generator: Faker,
    selected_entities: tuple[SourceEntity, ...],
    features: tuple[FeatureConfig, ...],
    repetition: int,
) -> tuple[dict[str, list], dict[int, list[str]], dict[int, list[str]]]:
    """Generate raw data rows with unique keys and shared IDs.

    This function generates rows of data plus maps between three types of identifiers:

        1. `id`: Is matchlab's unique identifier for each row, shared across rows with
            identical feature values
        2. `key`: Is the source's unique identifier for the row. It's like a primary key
            in a database, but not guaranteed to be unique across different entities
        3. `entity`: Is the identifier of the SourceEntity that generated the row.
            This identifies the true linked data in the factory system.

    This function will therefore return:

        * raw_data: A dictionary of column arrays for DataFrame creation
        * entity_keys: A dictionary that maps which keys belong to each source entity
        * id_keys: A dictionary that maps which keys share the same row content,
            with the same `id`

    The key insight:

        * entity_* groups by "who generated this row"
        * id_* groups by "what content does this row have"

    Example with two entities generating data:

    | id | key | company_name |
    |----|-----|--------------|
    | 1  | a   | alpha co     |
    | 2  | b   | alpha ltd    |
    | 1  | c   | alpha co     |  # Same content as row 'a'
    | 2  | d   | alpha ltd    |  # Same content as row 'b'
    | 3  | e   | beta co      |
    | 4  | f   | beta ltd     |
    | 3  | g   | beta co      |  # Same content as row 'e'
    | 4  | h   | beta ltd     |  # Same content as row 'f'

    What does this table look like as raw data?

    ```python
    raw_data = {
        "id": [1, 2, 1, 2, 3, 4, 3, 4],
        "key": ["a", "b", "c", "d", "e", "f", "g", "h"],
        "company_name": [
            "alpha co",
            "alpha ltd",
            "alpha co",
            "alpha ltd",
            "beta co",
            "beta ltd",
            "beta co",
            "beta ltd",
        ],
    }
    ```

    Which keys came from each source entity?

    ```python
    entity_keys = {
        1: ["a", "b", "c", "d"],  # All keys entity 1 produced
        2: ["e", "f", "g", "h"],  # All keys entity 2 produced
    }
    ```

    Which keys have identical content?

    ```python
    id_keys = {
        1: ["a", "c"],  # Both have "alpha co" content
        2: ["b", "d"],  # Both have "alpha ltd" content
        3: ["e", "g"],  # Both have "beta co" content
        4: ["f", "h"],  # Both have "beta ltd" content
    }
    ```
    """
    raw_data = {"key": [], "id": []}
    for feature in features:
        raw_data[feature.name] = []

    # Track entity locations and row identities
    entity_keys = {entity.id: [] for entity in selected_entities}
    id_keys = {}
    value_to_id = {}

    def add_row(entity_id: int, values: tuple) -> None:
        """Add a row of data, handling IDs and keys."""
        key = str(generator.uuid4())
        entity_keys[entity_id].append(key)

        if values not in value_to_id:
            mb_id = generator.random_number(digits=16)
            value_to_id[values] = mb_id
            id_keys[mb_id] = []

        row_id = value_to_id[values]
        id_keys[row_id].append(key)

        raw_data["key"].append(key)
        raw_data["id"].append(row_id)
        for feature, value in zip(features, values, strict=True):
            raw_data[feature.name].append(value)

    for entity in selected_entities:
        # For each feature, collect all possible values
        possible_values = []
        for feature in features:
            base = entity.base_values[feature.name]

            for rule in feature.variations:
                if not isinstance(base, rule.type):
                    raise TypeError(
                        f"{rule.__class__.__name__} requires {rule.type} "
                        f"but {feature.name} generates {type(base)}"
                    )

            variations = []
            # Apply all variations as long as they change the value
            for v in (rule.apply(base) for rule in feature.variations):
                if v != base:
                    variations.append(v)

            values = variations if feature.drop_base else variations + [base]
            possible_values.append(values or [base])

        if (num_variations := prod(len(values) for values in possible_values)) > 100:
            raise RuntimeError(
                f"Entity {entity.id} would generate {num_variations:,} variations, "
                "which exceeds the maximum of 100. This would cause a Cartesian "
                "explosion."
            )

        for values in product(*possible_values):
            for _ in range(repetition + 1):
                add_row(entity.id, values)

    return raw_data, entity_keys, id_keys


@cache
def generate_source(
    generator: Faker,
    n_true_entities: int,
    features: tuple[FeatureConfig, ...],
    repetition: int,
    seed_entities: tuple[SourceEntity, ...] | None = None,
) -> tuple[pa.Table, dict[int, set[str]], dict[int, set[str]]]:
    """Generate raw data as PyArrow tables with entity tracking.

    Returns:
        - data: PyArrow table with generated data
        - entity_keys: SourceEntity ID -> list of keys mapping
        - id_keys: Unique row ID -> list of keys mapping for identical rows
    """
    # Select or generate entities
    if seed_entities is None:
        selected_entities = generate_entities(generator, features, n_true_entities)
    else:
        selected_entities = generator.random_elements(
            elements=seed_entities,
            unique=True,
            length=min(n_true_entities, len(seed_entities)),
        )

    # Generate initial data
    raw_data, entity_keys, id_keys = generate_rows(
        generator=generator,
        selected_entities=selected_entities,
        features=features,
        repetition=repetition,
    )

    # Create DataFrame
    df = pd.DataFrame(raw_data)

    # Update variation counts
    for entity in selected_entities:
        if entity.id in entity_keys:
            # Count unique row IDs this entity appears in
            entity_rows = df[df["key"].isin(entity_keys[entity.id])]
            entity.total_unique_variations = len(set(entity_rows["id"]))

    return (
        pa.Table.from_pandas(df, preserve_index=False),
        entity_keys,
        id_keys,
    )


@make_features_hashable
@cache
def source_factory(
    features: list[FeatureConfig] | list[dict] | None = None,
    name: str | None = None,
    location_name: str = "dbname",
    engine: Engine | None = None,
    n_true_entities: int = 10,
    repetition: int = 0,
    seed: int = 42,
) -> SourceTestkit:
    """Generate a complete source testkit from configured features.

    Sources created with the factory system can only use a RelationalDBLocation,
    and the data at that location will be stored in a single table.

    Args:
        features: List of FeatureConfig objects or dictionaries to use for generating
            the source data. If None, defaults to a set of common features.
        name: Name of the source. If None, a unique name is generated. This will be
            used as the name of the table in the RelationalDBLocation, but also in
            the str for the source.
        location_name: Name of the location for the source.
        engine: SQLAlchemy engine to use for the source's RelationalDBLocation. If
            None, an in-memory SQLite engine is created.
        n_true_entities: Number of true entities to generate. Defaults to 10.
        repetition: Number of times to repeat the generated data. Defaults to 0.
        seed: Random seed for reproducibility. Defaults to 42.
    """
    generator = Faker()
    generator.seed_instance(seed)

    if features is None:
        features = (
            FeatureConfig(
                name="company_name",
                base_generator="company",
            ),
            FeatureConfig(
                name="crn",
                base_generator="bothify",
                parameters=(("text", "???-###-???-###"),),
            ),
        )

    if name is None:
        name = generator.unique.word()

    if engine is None:
        engine = create_engine("sqlite:///:memory:")

    # Generate base entities
    base_entities = generate_entities(
        generator=generator,
        features=features,
        n=n_true_entities,
    )

    # Generate data using the base entities
    data, entity_keys, row_keys = generate_source(
        generator=generator,
        n_true_entities=n_true_entities,
        features=features,
        repetition=repetition,
        seed_entities=base_entities,
    )

    # Create source entities with references
    source_entities = []
    for entity in base_entities:
        keys = entity_keys.get(entity.id, [])
        if keys:
            entity.add_source_reference(name, keys)
            source_entities.append(entity)

    # Create ClusterEntity objects from row_keys
    results_entities = [
        ClusterEntity(
            id=row_id,
            keys=EntityReference({name: frozenset(keys)}),
        )
        for row_id, keys in row_keys.items()
    ]

    # Create source config
    source = Source(
        location=RelationalDBLocation(name=location_name, client=engine),
        name=name,
        extract_transform=select(
            cast(column("key"), "string").as_("key"),
            *[column(feature.name) for feature in features],
        )
        .from_(name)
        .sql(),
        key_field="key",
    )

    return SourceTestkit(
        source=source,
        features=features,
        data=data,
        entities=tuple(sorted(results_entities)),
    )


def source_from_tuple(
    data_tuple: tuple[dict[str, Any], ...],
    data_keys: tuple[Any],
    name: str | None = None,
    location_name: str = "dbname",
    engine: Engine | None = None,
    seed: int = 42,
) -> SourceTestkit:
    """Generate a complete source testkit from dummy data."""
    generator = Faker()
    generator.seed_instance(seed)

    if name is None:
        name = generator.unique.word()

    if engine is None:
        engine = create_engine("sqlite:///:memory:")

    base_entities = tuple(SourceEntity(base_values=row) for row in data_tuple)

    # Create source entities with references
    source_entities: list[SourceEntity] = []
    for entity, key in zip(base_entities, data_keys, strict=True):
        entity.add_source_reference(name, [key])
        source_entities.append(entity)
    entity_ids = {entity.id for entity in source_entities}

    # Create ClusterEntity objects from row_keys
    results_entities = [
        ClusterEntity(
            id=entity_id,
            keys=EntityReference({name: frozenset([key])}),
        )
        for key, entity_id in zip(data_keys, entity_ids, strict=True)
    ]

    # Create source config
    source = Source(
        location=RelationalDBLocation(name=location_name, client=engine),
        name=name,
        extract_transform=select(
            cast(column("key"), "string").as_("key"),
            *[column(field) for field in data_tuple[0]],
        )
        .from_(name)
        .sql(),
        key_field="key",
    )

    raw_data = pa.Table.from_pylist(list(data_tuple))
    raw_keys = pa.array(data_keys)

    data = raw_data.append_column("id", [entity_ids]).append_column("key", raw_keys)

    return SourceTestkit(
        source=source,
        data=data,
        entities=tuple(sorted(results_entities)),
    )


@cache
def linked_sources_factory(
    source_parameters: tuple[SourceTestkitParameters, ...] | None = None,
    n_true_entities: int | None = None,
    engine: Engine | None = None,
    seed: int = 42,
) -> LinkedSourcesTestkit:
    """Generate a set of linked sources with tracked entities.

    Args:
        source_parameters: Optional tuple of source testkit parameters
        n_true_entities: Optional number of true entities to generate. If provided,
            overrides any n_true_entities in source configs. If not provided, each
            SourceTestkitParameters must specify its own n_true_entities.
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
            SourceTestkitParameters(
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
            SourceTestkitParameters(
                name="dh",
                engine=engine or default_engine,
                features=(
                    features["company_name"],
                    features["dh"],
                ),
                n_true_entities=n_true_entities // 2,
                repetition=0,
            ),
            SourceTestkitParameters(
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
            # Factory parameter provided - warn if configs have values set
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
            # No factory parameter - check all configs have n_true_entities set
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

    # Initialize LinkedSourcesTestkit
    true_entity_lookup = {entity.id: entity for entity in all_entities}
    linked = LinkedSourcesTestkit(
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

        # Create ClusterEntity objects from row_keys
        results_entities = [
            ClusterEntity(
                id=row_id,
                keys=EntityReference({parameters.name: frozenset(keys)}),
            )
            for row_id, keys in row_keys.items()
        ]

        # Create source config
        source = Source(
            location=RelationalDBLocation(
                name=str(parameters.name), client=parameters.engine
            ),
            name=parameters.name,
            extract_transform=select(
                cast(column("key"), "string").as_("key"),
                *[column(feature.name) for feature in parameters.features],
            )
            .from_(parameters.name)
            .sql(),
            key_field="key",
        )

        # Add source to linked.sources
        linked.sources[parameters.name] = SourceTestkit(
            source=source,
            features=tuple(parameters.features),
            data=data,
            entities=tuple(sorted(results_entities)),
        )

        # Update entities with source references
        for entity_id, keys in entity_keys.items():
            entity = true_entity_lookup[entity_id]
            entity.add_source_reference(parameters.name, keys)

    return linked
