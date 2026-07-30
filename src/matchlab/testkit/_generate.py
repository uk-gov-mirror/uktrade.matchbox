"""Row and entity generation.

Internal. Reach for `source_factory` or `linked_sources_factory` instead; these are the
pieces they are built from, and they take arguments that only make sense once the
callers have worked them out.
"""

from functools import cache
from itertools import product
from math import prod

import polars as pl
import pyarrow as pa
from faker import Faker

from matchlab.testkit.entities import EntityReference, TrueEntity
from matchlab.testkit.features import FeatureConfig


@cache
def generate_entities(
    generator: Faker,
    features: tuple[FeatureConfig, ...],
    n: int,
) -> tuple[TrueEntity]:
    """Generate base entities with their ground truth values from generator."""
    entities = []
    for _ in range(n):
        base_values = {}
        for feature in features:
            generator_func = generator.unique if feature.unique else generator
            value_generator = getattr(generator_func, feature.base_generator)
            parameters = {} if not feature.parameters else dict(feature.parameters)

            value = value_generator(**parameters)
            # Explicitly cast lists to tuples to ensure they are hashable
            if isinstance(value, list):
                value = tuple(value)
            base_values[feature.name] = value

        entities.append(TrueEntity(base_values=base_values, keys=EntityReference()))
    return tuple(entities)


def generate_rows(
    generator: Faker,
    selected_entities: tuple[TrueEntity, ...],
    features: tuple[FeatureConfig, ...],
    repetition: int,
) -> tuple[dict[str, list], dict[int, list[str]], dict[int, list[str]]]:
    """Generate raw data rows with unique keys and shared IDs.

    This function generates rows of data plus maps between three types of identifiers:

        1. `id`: Is matchlab's unique identifier for each row, shared across rows with
            identical feature values
        2. `key`: Is the source's unique identifier for the row. It's like a primary key
            in a database, but not guaranteed to be unique across different entities
        3. `entity`: Is the identifier of the TrueEntity that generated the row.
            This identifies the true linked data in the factory system.

    This function will therefore return:

        * raw_data: A dictionary of column arrays for DataFrame creation
        * entity_keys: A dictionary that maps which keys belong to each true entity
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

    Which keys came from each true entity?

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
    seed_entities: tuple[TrueEntity, ...] | None = None,
) -> tuple[pa.Table, dict[int, set[str]], dict[int, set[str]]]:
    """Generate raw data as PyArrow tables with entity tracking.

    Note: this is `@cache`d but also *mutates* `total_unique_variations` on the shared
    `TrueEntity` objects it is given, so a cache hit skips that mutation. Harmless
    today because the first call for a given key does the writing, but it means the
    function is not free of side effects and the cache cannot simply be dropped.

    Returns:
        - data: PyArrow table with generated data
        - entity_keys: TrueEntity ID -> list of keys mapping
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
    df = pl.DataFrame(raw_data)

    # Update variation counts
    for entity in selected_entities:
        if entity.id in entity_keys:
            # Count unique row IDs this entity appears in
            entity_rows = df.filter(pl.col("key").is_in(list(entity_keys[entity.id])))
            entity.total_unique_variations = entity_rows["id"].n_unique()

    return (
        df.to_arrow(),
        entity_keys,
        id_keys,
    )
