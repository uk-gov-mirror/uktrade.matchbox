"""Tests for generating synthetic sources and their rows."""

import functools

import pytest
from faker import Faker
from sqlalchemy import Engine

from matchlab.sources import RelationalDB
from matchlab.testkit._generate import generate_rows
from matchlab.testkit.entities import TrueEntity
from matchlab.testkit.features import FeatureConfig, ReplaceRule, SuffixRule
from matchlab.testkit.sources import source_factory, source_from_tuple, values_of


def test_source_factory_default() -> None:
    """The default factory produces ten rows, each with its own id."""
    source = source_factory()

    assert len(source.input_clusters) == 10
    assert source.data.shape[0] == 10
    assert len(set(source.data.column("id").to_pylist())) == 10


def test_source_factory_repetition() -> None:
    """Repetition adds that many extra keys sharing each row's id and feature values."""
    features = [
        FeatureConfig(
            name="company_name",
            base_generator="company",
            variations=[SuffixRule(suffix=" Inc")],
        ),
    ]

    n_true_entities = 2
    repetition = 3
    source = source_factory(
        n_true_entities=n_true_entities,
        repetition=repetition,
        features=features,
        seed=42,
    )

    data_df = source.data.to_pandas()

    # Each `id` groups the rows sharing identical content
    for _, rows in data_df.groupby("id"):
        # repetition + 1 keys per id: the base row plus its repeats
        keys = rows["key"]
        assert len(keys) == len(keys.unique())
        assert len(keys) == repetition + 1

        for feature in features:
            assert len(rows[feature.name].unique()) == 1

    # One id per entity per variation, including the base value
    expected_unique_rows = n_true_entities * (len(features[0].variations) + 1)
    assert len(data_df["id"].unique()) == expected_unique_rows

    assert len(data_df) == expected_unique_rows * (repetition + 1)


def test_source_factory_row_id_integrity() -> None:
    """Rows sharing an id always share identical feature values, and vice versa."""
    features = [
        FeatureConfig(
            name="company_name",
            base_generator="company",
            variations=[SuffixRule(suffix=" Inc")],
        ),
    ]

    n_true_entities = 3
    repetition = 1
    source_testkit = source_factory(
        n_true_entities=n_true_entities,
        repetition=repetition,
        features=features,
        seed=42,
    )

    data_df = source_testkit.data.to_pandas()

    for _, rows in data_df.groupby("id"):
        for feature in features:
            assert len(rows[feature.name].unique()) == 1

    # repetition=1 means each unique row appears in one id group of two keys
    assert all(len(rows["key"]) == repetition + 1 for _, rows in data_df.groupby("id"))

    # One id group per unique row, per entity
    expected_id_groups = n_true_entities * (
        len(features[0].variations) + 1
    )  # +1 for base value
    assert len(data_df["id"].unique()) == expected_id_groups


def test_source_factory_mock_properties(sqlite_in_memory_warehouse: Engine) -> None:
    """A source built by source_factory carries its features, name, and key field."""
    features = [
        FeatureConfig(
            name="company_name",
            base_generator="company",
            variations=(SuffixRule(suffix=" Ltd"),),
        ),
        FeatureConfig(
            name="registration_id",
            base_generator="numerify",
            parameters=(("text", "######"),),
        ),
    ]

    name = "companies"
    location_name = "custom_name"

    source_testkit = source_factory(
        features=features,
        name=name,
        location_name=location_name,
        engine=sqlite_in_memory_warehouse,
    )
    source_spec = source_testkit.source.spec

    assert source_testkit.source.location_resources["client"].name == location_name

    # Every generated feature is selected by the query, so it's part of identity.
    location = source_testkit.source.location
    assert isinstance(location, RelationalDB)
    assert source_testkit.field_names == [feature.name for feature in features]
    for feature in features:
        assert feature.name in location.sql

    assert source_testkit.source.name == name
    assert source_spec.key_field == "key"

    dump = source_spec.model_dump()
    assert dump["key_field"] == "key"
    assert "index_fields" not in dump


def test_entity_variations_tracking() -> None:
    """Each Cluster from source_factory is a proper subset of its parent entity."""
    features = [
        FeatureConfig(
            name="company",
            base_generator="company",
            variations=[
                SuffixRule(suffix=" Inc"),
                SuffixRule(suffix=" Ltd"),
            ],
            drop_base=True,
        )
    ]

    source_testkit = source_factory(features=features, n_true_entities=2, seed=42)

    for cluster_entity in source_testkit.input_clusters:
        entity_values = values_of(
            cluster_entity, {source_testkit.source.name: source_testkit}
        )

        unique_variations = 0
        for features_values in entity_values.values():
            for values in features_values.values():
                unique_variations += len(values)

        # drop_base=True drops the base value, leaving one variation per cluster
        assert unique_variations == 1

        data_df = source_testkit.data.to_pandas()

        result_keys = cluster_entity.get_keys(source_testkit.source.name)
        assert result_keys is not None

        # Every row in one cluster shares the same company value
        result_rows = data_df[data_df["key"].isin(result_keys)]
        assert len(result_rows["company"].unique()) == 1

        company_values = result_rows["company"]
        # drop_base=True means only the suffixed variations ever appear
        assert all(
            value.endswith(" Inc") or value.endswith(" Ltd") for value in company_values
        )


def test_base_and_variation_entities() -> None:
    """Keeping the base value alongside one variation makes two separate clusters."""
    features = [
        FeatureConfig(
            name="company",
            base_generator="company",
            variations=[SuffixRule(suffix=" Inc")],
            drop_base=False,  # Keep base value
        )
    ]

    source_testkit = source_factory(features=features, n_true_entities=1, seed=42)

    assert len(source_testkit.input_clusters) == 2

    data_df = source_testkit.data.to_pandas()

    all_company_values = data_df["company"].unique().tolist()

    # The base value is whichever one doesn't carry the " Inc" suffix
    base_value = next(
        value for value in all_company_values if not value.endswith(" Inc")
    )
    variation_value = next(
        value for value in all_company_values if value.endswith(" Inc")
    )

    base_entity = None
    variation_entity = None

    for entity in source_testkit.input_clusters:
        entity_keys = entity.get_keys(source_testkit.source.name)
        rows = data_df[data_df["key"].isin(entity_keys)]
        values = rows["company"]
        assert len(values.unique()) == 1
        value = values.iloc[0]

        if value == base_value:
            base_entity = entity
        elif value == variation_value:
            variation_entity = entity

    assert base_entity is not None, "No Cluster found for base value"
    assert variation_entity is not None, "No Cluster found for variation"

    # Each cluster holds only its own variation, never the other one
    base_values = values_of(base_entity, {source_testkit.source.name: source_testkit})
    assert base_values[source_testkit.source.name]["company"] == [base_value]

    variation_values = values_of(
        variation_entity, {source_testkit.source.name: source_testkit}
    )
    assert variation_values[source_testkit.source.name]["company"] == [variation_value]

    # Adding the two clusters back together should recompose the full entity
    combined = base_entity + variation_entity

    combined_values = values_of(combined, {source_testkit.source.name: source_testkit})
    assert sorted(combined_values[source_testkit.source.name]["company"]) == sorted(
        [base_value, variation_value]
    )

    assert (
        combined.keys[source_testkit.source.name]
        == base_entity.keys[source_testkit.source.name]
        | variation_entity.keys[source_testkit.source.name]
    )

    # Diffing either cluster against the other returns exactly its own keys
    base_diff = base_entity - variation_entity
    assert (
        base_diff.get(source_testkit.source.name)
        == base_entity.keys[source_testkit.source.name]
    )

    variation_diff = variation_entity - base_entity
    assert (
        variation_diff.get(source_testkit.source.name)
        == variation_entity.keys[source_testkit.source.name]
    )


def test_source_factory_id_generation() -> None:
    """Rows share an id exactly when their feature values match, and ids are int64."""
    features = [
        FeatureConfig(
            name="company_name",
            base_generator="company",
            variations=[SuffixRule(suffix=" Inc")],
        ),
    ]

    n_true_entities = 2
    repetition = 2
    source = source_factory(
        n_true_entities=n_true_entities,
        repetition=repetition,
        features=features,
        seed=42,
    )

    data_df = source.data.to_pandas()

    for _, group in data_df.groupby("company_name"):
        assert len(group["id"].unique()) == 1

    assert data_df["id"].dtype == "int64"

    assert len(data_df["id"].unique()) == len(data_df["company_name"].unique())


def test_source_from_tuple() -> None:
    """source_from_tuple builds a testkit directly from rows and keys, no generation."""
    data_tuple = ({"a": 1, "b": "val"}, {"a": 2, "b": "val"})
    testkit = source_from_tuple(data_tuple=data_tuple, data_keys=["0", "1"], name="foo")

    assert len(testkit.input_clusters) == 2
    assert set(testkit.input_clusters[0].keys["foo"]) | set(
        testkit.input_clusters[1].keys["foo"]
    ) == {"0", "1"}

    assert testkit.data.shape[0] == 2
    assert set(testkit.data.column_names) == {"id", "key", "a", "b"}
    assert len(set(testkit.data.column("id").to_pylist())) == 2
    assert set(testkit.field_names) == {"a", "b"}


@pytest.mark.parametrize(
    ("selected_entities", "features", "repetition"),
    [
        pytest.param(
            # Base case: Two entities, one feature, unique values, no repetition
            (
                TrueEntity(base_values={"name": "alpha"}),
                TrueEntity(base_values={"name": "beta"}),
            ),
            (FeatureConfig(name="name", base_generator="name"),),
            0,
            id="two_entities_unique_values_no_repetition",
        ),
        pytest.param(
            # Same case with repetition
            (
                TrueEntity(base_values={"name": "alpha"}),
                TrueEntity(base_values={"name": "beta"}),
            ),
            (FeatureConfig(name="name", base_generator="name"),),
            3,  # 3 repetitions
            id="two_entities_unique_values_with_repetition",
        ),
        pytest.param(
            # Case: two entities with identical values share IDs, two repetitions
            (
                TrueEntity(base_values={"name": "alpha"}),
                TrueEntity(base_values={"name": "alpha"}),
            ),
            (FeatureConfig(name="name", base_generator="name"),),
            2,
            id="two_entities_same_values_with_repetition",
        ),
        pytest.param(
            # Case: Multiple features, tests tuple-based identity, one repetition
            (
                TrueEntity(base_values={"name": "alpha", "user_id": "123"}),
                TrueEntity(base_values={"name": "alpha", "user_id": "456"}),
            ),
            (
                FeatureConfig(name="name", base_generator="name"),
                FeatureConfig(name="user_id", base_generator="uuid4"),
            ),
            1,
            id="multiple_features_partial_match_with_repetition",
        ),
        pytest.param(
            # Case: an empty entities list is handled gracefully, even with repetition
            (),
            (FeatureConfig(name="name", base_generator="name"),),
            5,
            id="empty_entities_with_repetition",
        ),
        pytest.param(
            # Case: Entity with variations and drop_base, two repetitions
            (TrueEntity(base_values={"name": "alpha"}),),
            (
                FeatureConfig(
                    name="name",
                    base_generator="name",
                    drop_base=True,
                    variations=(
                        ReplaceRule(old="a", new="@"),  # alpha -> @lph@
                        ReplaceRule(old="a", new="4"),  # alpha -> 4lph4
                    ),
                ),
            ),
            2,
            id="variations_with_drop_base_and_repetition",
        ),
        pytest.param(
            # Case: Entity with variations and drop_base, one repetition
            (TrueEntity(base_values={"name": "alpha", "user_id": "123"}),),
            (
                FeatureConfig(
                    name="name",
                    base_generator="name",
                    drop_base=True,
                    variations=(
                        ReplaceRule(old="a", new="@"),  # alpha -> @lph@
                        ReplaceRule(old="a", new="4"),  # alpha -> 4lph4
                    ),
                ),
                FeatureConfig(
                    name="user_id",
                    base_generator="uuid4",
                    # No variations, keeps base value
                ),
            ),
            1,
            id="mixed_variations_and_drop_base_with_repetition",
        ),
        pytest.param(
            # Case: Multiple entities with mixed variation configs, four repetitions
            (
                TrueEntity(base_values={"name": "alpha", "title": "ceo"}),
                TrueEntity(base_values={"name": "beta", "title": "cto"}),
            ),
            (
                FeatureConfig(
                    name="name",
                    base_generator="name",
                    drop_base=False,  # Keeps original
                    variations=(ReplaceRule(old="a", new="@"),),
                ),
                FeatureConfig(
                    name="title",
                    base_generator="job",
                    drop_base=True,  # Drops original
                    variations=(
                        ReplaceRule(old="o", new="0"),
                        ReplaceRule(old="e", new="3"),  # Won't affect CTO
                    ),
                ),
            ),
            4,
            id="multiple_entities_mixed_variations_with_repetition",
        ),
    ],
)
def test_generate_rows(
    selected_entities: tuple[TrueEntity, ...],
    features: tuple[FeatureConfig, ...],
    repetition: int,
) -> None:
    """generate_rows tracks which keys belong to each entity and share a row id."""
    generator = Faker(seed=42)
    raw_data, entity_keys, id_keys = generate_rows(
        generator=generator,
        selected_entities=selected_entities,
        features=features,
        repetition=repetition,
    )

    n_rows = len(raw_data["key"])
    assert len(raw_data["id"]) == n_rows
    assert all(len(values) == n_rows for values in raw_data.values())

    assert len(selected_entities) == len(entity_keys)

    # Each unique combination of feature values gets exactly one id
    unique_values = {
        tuple(raw_data[f.name][i] for f in features) for i in range(n_rows)
    }
    assert len(unique_values) == len(id_keys)

    if repetition > 0:
        # repetition adds extra keys per id, never duplicate ones
        for id_group_keys in id_keys.values():
            assert len(id_group_keys) == len(set(id_group_keys))

    value_counts = {}
    for i in range(n_rows):
        values = tuple(raw_data[f.name][i] for f in features)
        row_id = raw_data["id"][i]
        value_counts[values] = value_counts.get(values, 0) + 1

    # An id's key count must match how often its values occur
    for i in range(n_rows):
        values = tuple(raw_data[f.name][i] for f in features)
        row_id = raw_data["id"][i]
        assert len(id_keys[row_id]) == value_counts[values]

    all_keys = set(raw_data["key"])
    assert all(key in all_keys for keys in entity_keys.values() for key in keys)
    assert all(key in all_keys for keys in id_keys.values() for key in keys)

    if not selected_entities:
        assert not raw_data["key"]
        assert not entity_keys
        assert not id_keys

    for entity in selected_entities:
        entity_rows = {
            i for i, key in enumerate(raw_data["key"]) if key in entity_keys[entity.id]
        }

        for feature in features:
            values = {raw_data[feature.name][i] for i in entity_rows}
            base_value = entity.base_values[feature.name]

            if feature.drop_base:
                assert base_value not in values
            elif not feature.variations:
                assert values == {base_value}

            # A rule that leaves the value unchanged doesn't count as a variation
            effective_variations = [
                rule.apply(base_value)
                for rule in feature.variations
                if rule.apply(base_value) != base_value
            ]

            if feature.variations:
                expected_count = len(effective_variations) + (
                    0 if feature.drop_base else 1
                )
                assert len(values) == expected_count

    for entity in selected_entities:
        variation_counts = []
        for feature in features:
            base_value = entity.base_values[feature.name]
            effective_variations = [
                rule.apply(base_value)
                for rule in feature.variations
                if rule.apply(base_value) != base_value
            ]
            # Each feature contributes its variations, plus the base value if kept
            if feature.drop_base and effective_variations:
                variation_counts.append(len(effective_variations))
            else:
                variation_counts.append(len(effective_variations) + 1)

        # Every combination across features, repeated (repetition + 1) times
        expected_unique_combinations = functools.reduce(
            lambda x, y: x * y, variation_counts, 1
        )
        expected_total_rows = expected_unique_combinations * (repetition + 1)
        assert len(entity_keys[entity.id]) == expected_total_rows

    assert set(id_keys.keys()) == set(raw_data["id"])

    values_to_id = {}
    for i in range(n_rows):
        values = tuple(raw_data[f.name][i] for f in features)
        row_id = raw_data["id"][i]

        if values not in values_to_id:
            values_to_id[values] = row_id
        else:
            # Repeat occurrences of the same values must map to the same id
            assert values_to_id[values] == row_id

    if len(unique_values) > 1:
        assert len(set(values_to_id.values())) == len(unique_values)
