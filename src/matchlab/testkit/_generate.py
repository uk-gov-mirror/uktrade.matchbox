"""Row and entity generation.

Internal. Reach for `source_factory` or `linked_sources_factory` instead. These are the
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
    """Generate base entities with their ground truth values from the generator."""
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
    """Generate raw data rows, plus the maps that connect three different IDs.

    Every row carries three IDs, each answering a different question. Confusing them is
    the mistake this docstring exists to prevent.

    * `key`: the row's own identifier. Like a source's primary key, unique within the
      source but not across entities.
    * `id`: matchlab's identifier for the row's *content*. Rows with identical feature
      values share one `id`, whichever entity produced them.
    * `entity`: the ID of the `TrueEntity` that generated the row. This is the planted
      answer, and the one truth downstream tests check their result against.

    `entity` groups rows by who generated them, and `id` groups rows by what content
    they hold. The two groupings diverge whenever an entity produces more than one
    variation: rows from the same entity can carry different `id`s, and (via
    `repetition`) the same `id` can appear on more than one row.

    Returns:
        - raw_data: column arrays, ready to build a DataFrame from
        - entity_keys: `TrueEntity` ID -> the keys it produced
        - id_keys: content `id` -> the keys sharing that content

    Worked example, two entities, each producing two variations twice over:

    | id | key | company_name |
    |----|-----|--------------|
    | 1  | a   | alpha co     |
    | 2  | b   | alpha ltd    |
    | 1  | c   | alpha co     |  # same content as row a
    | 2  | d   | alpha ltd    |  # same content as row b
    | 3  | e   | beta co      |
    | 4  | f   | beta ltd     |
    | 3  | g   | beta co      |  # same content as row e
    | 4  | h   | beta ltd     |  # same content as row f

    As raw data:

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

    Grouped by entity, the two variations sit together:

    ```python
    entity_keys = {
        1: ["a", "b", "c", "d"],  # every key entity 1 produced
        2: ["e", "f", "g", "h"],  # every key entity 2 produced
    }
    ```

    Grouped by content, they split apart again:

    ```python
    id_keys = {
        1: ["a", "c"],  # both rows read "alpha co"
        2: ["b", "d"],  # both rows read "alpha ltd"
        3: ["e", "g"],  # both rows read "beta co"
        4: ["f", "h"],  # both rows read "beta ltd"
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
        # Collect all possible values, per feature.
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
