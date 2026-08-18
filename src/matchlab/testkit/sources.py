"""One generated source: a real plan node, its rows, and the partition they imply."""

from collections.abc import Callable
from functools import cache, wraps
from typing import Any, ParamSpec, Self, TypeVar

import polars as pl
import pyarrow as pa
from faker import Faker
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Engine, create_engine
from sqlglot import cast, select
from sqlglot.expressions import column

from matchlab.resources import Resource
from matchlab.sources import RelationalDB, Source, read_database
from matchlab.testkit._generate import generate_entities, generate_source
from matchlab.testkit.entities import Cluster, EntityReference, TrueEntity
from matchlab.testkit.features import FeatureConfig

P = ParamSpec("P")
R = TypeVar("R")


def make_features_hashable(func: Callable[P, R]) -> Callable[P, R]:
    """Let callers configure `source_factory` with plain dicts.

    Converts each dict to a `FeatureConfig` before the wrapped function runs, so
    `source_factory` stays hashable, which `@cache` needs, while its callers don't have
    to build `FeatureConfig` objects by hand.
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


class GeneratedSource(BaseModel):
    """Generated rows, and the `Source` plan node that reads them.

    Sources that share entities are what make linking testable. Use
    `linked_sources_factory` in `linked` instead when you need that.

    This exposes what it knows about the *fixture*: the rows, the features they were
    generated from, and the partition they imply. Anything you do to the *plan* goes
    through `.source`, which is the node itself: `source.source.clean(...)`,
    `source.source.name`, `source.source.spec`. There is deliberately no shortcut. A
    testkit that forwarded those would be indistinguishable from the node it wraps, and
    knowing which one you are holding is the whole point.
    """

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
    input_clusters: tuple[Cluster, ...] = Field(
        description=(
            "The partition a matcher starts from: generated rows grouped by identical "
            "content. Not the answer, that is LinkedSources.true_entities."
        )
    )

    @property
    def field_names(self) -> list[str]:
        """The non-key columns of this testkit's data, in generation order.

        Taken from the testkit rather than from `Source.index_fields`, which would have
        to read the warehouse. These names are needed before a plan is even built.
        Falls back to the data's own columns when the source was set manually rather
        than generated from features.
        """
        if self.features is not None:
            return [feature.name for feature in self.features]
        return [
            column for column in self.data.column_names if column not in ("key", "id")
        ]

    def write_to_location(self) -> Self:
        """Write the data to the source's location.

        Pass `engine` to `source_factory` to choose which warehouse that is. Testkits
        are generated against an in-memory SQLite engine otherwise.

        Returns:
            This testkit.
        """
        if not isinstance(self.source.location, RelationalDB):
            raise TypeError(
                "A testkit writes its rows to a relational warehouse, so its source "
                f"must read one. This one reads a "
                f"{type(self.source.location).__name__}."
            )

        pl.from_arrow(self.data).write_database(
            table_name=self.source.name,
            connection=self.source.location.client,
            if_table_exists="replace",
        )

        return self


def values_of(
    entity: TrueEntity | Cluster, sources: dict[str, GeneratedSource]
) -> dict[str, dict[str, list[str]]]:
    """Every unique value this entity took, per source and per feature.

    Args:
        entity: The entity whose rows to look up.
        sources: The generated sources to look in, by name.

    Returns:
        `{source: {feature: sorted unique values}}`. Each source may vary the same base
        value differently, so they are kept apart.

    Raises:
        ValueError: If the entity references a source that was not supplied.
    """
    values: dict[str, dict[str, list[str]]] = {}

    for source_name, keys in entity.keys.items():
        source = sources.get(source_name)
        if source is None:
            raise ValueError(f"Source testkit not found: {source_name}")

        rows = pl.DataFrame(source.data).filter(pl.col("key").is_in(list(keys)))
        values[source_name] = {
            feature.name: sorted(rows[feature.name].unique().to_list())
            for feature in source.features
        }

    return values


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
) -> GeneratedSource:
    """Generate a complete source testkit from configured features.

    Sources created with the factory system can only use a RelationalDB,
    and the data at that location will be stored in a single table.

    Args:
        features: List of FeatureConfig objects or dictionaries to use for generating
            the source data. If None, defaults to a set of common features.
        name: Name of the source. If None, a unique name is generated. This will be
            used as the name of the table in the RelationalDB, but also in
            the str for the source.
        location_name: The param name the engine is bound to, which is what a dumped
            plan names and `load` must supply.
        engine: SQLAlchemy engine to use for the source's RelationalDB. If
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

    # Create true entities with references
    true_entities = []
    for entity in base_entities:
        keys = entity_keys.get(entity.id, [])
        if keys:
            entity.add_source_reference(name, keys)
            true_entities.append(entity)

    # Create Cluster objects from row_keys
    clusters = [
        Cluster(
            id=row_id,
            keys=EntityReference({name: frozenset(keys)}),
        )
        for row_id, keys in row_keys.items()
    ]

    # Create source config
    source = read_database(
        name,
        sql=select(
            cast(column("key"), "string").as_("key"),
            *[column(feature.name) for feature in features],
        )
        .from_(name)
        .sql(),
        client=Resource(location_name, engine),
        key_field="key",
    )

    return GeneratedSource(
        source=source,
        features=features,
        data=data,
        input_clusters=tuple(sorted(clusters)),
    )


def source_from_tuple(
    data_tuple: tuple[dict[str, Any], ...],
    data_keys: tuple[Any],
    name: str | None = None,
    location_name: str = "dbname",
    engine: Engine | None = None,
    seed: int = 42,
) -> GeneratedSource:
    """Generate a complete source testkit from dummy data."""
    generator = Faker()
    generator.seed_instance(seed)

    if name is None:
        name = generator.unique.word()

    if engine is None:
        engine = create_engine("sqlite:///:memory:")

    base_entities = tuple(TrueEntity(base_values=row) for row in data_tuple)

    # Create true entities with references
    true_entities: list[TrueEntity] = []
    for entity, key in zip(base_entities, data_keys, strict=True):
        entity.add_source_reference(name, [key])
        true_entities.append(entity)
    # A list, not a set: these are zipped against `data_keys` below, so they must stay
    # in row order. A set comprehension paired each key with an arbitrary entity.
    entity_ids = [entity.id for entity in true_entities]

    # Create Cluster objects from row_keys
    clusters = [
        Cluster(
            id=entity_id,
            keys=EntityReference({name: frozenset([key])}),
        )
        for key, entity_id in zip(data_keys, entity_ids, strict=True)
    ]

    # Create source config
    source = read_database(
        name,
        sql=select(
            cast(column("key"), "string").as_("key"),
            *[column(field) for field in data_tuple[0]],
        )
        .from_(name)
        .sql(),
        client=Resource(location_name, engine),
        key_field="key",
    )

    raw_data = pa.Table.from_pylist(list(data_tuple))
    raw_keys = pa.array(data_keys)

    data = raw_data.append_column("id", [entity_ids]).append_column("key", raw_keys)

    return GeneratedSource(
        source=source,
        data=data,
        input_clusters=tuple(sorted(clusters)),
    )
