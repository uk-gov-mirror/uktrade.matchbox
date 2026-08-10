"""Tests for generating synthetic sources and their rows."""

import functools

import pytest
from faker import Faker
from sqlalchemy import Engine

from matchlab.testkit._generate import generate_rows
from matchlab.testkit.entities import TrueEntity
from matchlab.testkit.features import FeatureConfig, ReplaceRule, SuffixRule
from matchlab.testkit.sources import source_factory, source_from_tuple, values_of


def test_source_factory_default() -> None:
    """Test that source_factory generates a source testkit with default parameters."""
    source = source_factory()

    assert len(source.input_clusters) == 10
    assert source.data.shape[0] == 10
    assert len(set(source.data.column("id").to_pylist())) == 10


def test_source_factory_repetition() -> None:
    """Test that source_factory correctly handles row repetition."""
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

    # Convert to pandas for easier analysis
    data_df = source.data.to_pandas()

    # Each `id` groups the rows sharing identical content
    for _, rows in data_df.groupby("id"):
        # Each id should have repetition + 1 (base) number of keys
        keys = rows["key"]
        assert len(keys) == len(keys.unique())
        assert len(keys) == repetition + 1

        # All rows should have identical feature values
        for feature in features:
            assert len(rows[feature.name].unique()) == 1

    # Total number of unique ids should be n_true_entities * (variations + 1)
    expected_unique_rows = n_true_entities * (len(features[0].variations) + 1)
    assert len(data_df["id"].unique()) == expected_unique_rows

    # Total number of rows should be unique ids * repetition + 1 (base)
    assert len(data_df) == expected_unique_rows * (repetition + 1)


def test_source_factory_row_id_integrity() -> None:
    """Test that row `id`s correctly identify identical rows."""
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

    # Convert to pandas for easier analysis
    data_df = source_testkit.data.to_pandas()

    # For each id group, verify that the corresponding rows are identical
    for _, rows in data_df.groupby("id"):
        # All rows sharing an id should have identical feature values
        for feature in features:
            assert len(rows[feature.name].unique()) == 1

    # Due to repetition=1, each unique row should appear
    # in exactly one id group with two keys
    # Repetition + 1 because we include the base value
    assert all(len(rows["key"]) == repetition + 1 for _, rows in data_df.groupby("id"))

    # Total number of id groups should equal
    # number of unique rows * number of true entities
    expected_id_groups = n_true_entities * (
        len(features[0].variations) + 1
    )  # +1 for base value
    assert len(data_df["id"].unique()) == expected_id_groups


def test_source_factory_mock_properties(sqlite_in_memory_warehouse: Engine) -> None:
    """Test that properties set in source_factory match generated SourceSpec."""
    # Create source with specific features and name
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

    assert source_testkit.source.location.name == location_name

    # Every generated feature is selected by the extract, and so is part of identity
    assert source_testkit.field_names == [feature.name for feature in features]
    for feature in features:
        assert feature.name in source_spec.extract_transform

    # Check default step name and default key field
    assert source_testkit.source.name == name
    assert source_spec.key_field == "key"

    # Verify source properties are preserved through model_dump
    dump = source_spec.model_dump()
    assert dump["key_field"] == "key"
    assert "index_fields" not in dump


def test_entity_variations_tracking() -> None:
    """Test that entity variations are correctly tracked and accessible.

    Asserts that Cluster objects are proper subsets of their parent entities.
    """
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

    # Process each Cluster group
    for cluster_entity in source_testkit.input_clusters:
        # Get the values for this entity
        entity_values = values_of(
            cluster_entity, {source_testkit.source.name: source_testkit}
        )

        # Calculate total unique variations (equivalent to total_unique_variations)
        unique_variations = 0
        for features_values in entity_values.values():
            for values in features_values.values():
                unique_variations += len(values)

        # With drop_base=True, we should only have the non-drop variations
        # Each entity should have exactly one variation
        assert unique_variations == 1

        # Verify the data values match expectations
        data_df = source_testkit.data.to_pandas()

        # Get keys for this cluster
        result_keys = cluster_entity.get_keys(source_testkit.source.name)
        assert result_keys is not None

        # All rows for a given cluster should share the same company value
        result_rows = data_df[data_df["key"].isin(result_keys)]
        assert len(result_rows["company"].unique()) == 1

        company_values = result_rows["company"]
        # With drop_base=True, should only see variation values
        assert all(
            value.endswith(" Inc") or value.endswith(" Ltd") for value in company_values
        )


def test_base_and_variation_entities() -> None:
    """Test that base values and variations create correct Cluster objects."""
    features = [
        FeatureConfig(
            name="company",
            base_generator="company",
            variations=[SuffixRule(suffix=" Inc")],
            drop_base=False,  # Keep base value
        )
    ]

    source_testkit = source_factory(features=features, n_true_entities=1, seed=42)

    # Should have two Cluster objects - one for base, one for variation
    assert len(source_testkit.input_clusters) == 2

    # Get the base and variation entities
    data_df = source_testkit.data.to_pandas()

    # We'll need to find the base value by examining the data
    # Get all unique company values
    all_company_values = data_df["company"].unique().tolist()

    # Identify which value is the base (doesn't end with " Inc")
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

    # Verify that each entity only contains its own variation
    base_values = values_of(base_entity, {source_testkit.source.name: source_testkit})
    assert base_values[source_testkit.source.name]["company"] == [base_value]

    variation_values = values_of(
        variation_entity, {source_testkit.source.name: source_testkit}
    )
    assert variation_values[source_testkit.source.name]["company"] == [variation_value]

    # Together they should compose the full set of entity data
    combined = base_entity + variation_entity

    # Verify that the combined entity contains both variations
    combined_values = values_of(combined, {source_testkit.source.name: source_testkit})
    assert sorted(combined_values[source_testkit.source.name]["company"]) == sorted(
        [base_value, variation_value]
    )

    # Verify that adding the entities produces the same result as having all keys
    assert (
        combined.keys[source_testkit.source.name]
        == base_entity.keys[source_testkit.source.name]
        | variation_entity.keys[source_testkit.source.name]
    )

    # The diff between entities should match their respective keys
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
    """Test that source_factory generates unique IDs for rows."""
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

    # Convert to pandas for easier analysis
    data_df = source.data.to_pandas()

    # Each unique row combination (excluding key) should get a different ID
    for _, group in data_df.groupby("company_name"):
        # All rows with same features should have same ID
        assert len(group["id"].unique()) == 1

    # Verify we're generating int64 IDs
    assert data_df["id"].dtype == "int64"

    # Different rows should have different IDs
    assert len(data_df["id"].unique()) == len(data_df["company_name"].unique())


def test_source_from_tuple() -> None:
    """Test that source_factory can create a source from a tuple of values."""
    # Create a source from a tuple of values
    data_tuple = ({"a": 1, "b": "val"}, {"a": 2, "b": "val"})
    testkit = source_from_tuple(data_tuple=data_tuple, data_keys=["0", "1"], name="foo")

    # Verify the generated testkit has the expected properties
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
            # Case: Two entities with identical values - should share IDs,
            # two repetitions
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
            # Case: Empty entities list - should handle gracefully, even with repetition
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
    """Test generate_rows correctly tracks entities and row identities."""
    generator = Faker(seed=42)
    raw_data, entity_keys, id_keys = generate_rows(
        generator=generator,
        selected_entities=selected_entities,
        features=features,
        repetition=repetition,
    )

    # Check arrays have consistent lengths
    n_rows = len(raw_data["key"])
    assert len(raw_data["id"]) == n_rows
    assert all(len(values) == n_rows for values in raw_data.values())

    # Check entity tracking - each entity appears exactly once
    assert len(selected_entities) == len(entity_keys)

    # Check row identity tracking - each unique value combo gets one ID
    unique_values = {
        tuple(raw_data[f.name][i] for f in features) for i in range(n_rows)
    }
    assert len(unique_values) == len(id_keys)

    # Test repetition behavior
    if repetition > 0:
        # All keys in each ID group should be unique (no duplicate keys)
        for id_group_keys in id_keys.values():
            assert len(id_group_keys) == len(set(id_group_keys))

    # When we have duplicate values, verify correct ID sharing
    value_counts = {}
    for i in range(n_rows):
        values = tuple(raw_data[f.name][i] for f in features)
        row_id = raw_data["id"][i]
        value_counts[values] = value_counts.get(values, 0) + 1

    # Each ID's keys set should match the number of times those values appear
    for i in range(n_rows):
        values = tuple(raw_data[f.name][i] for f in features)
        row_id = raw_data["id"][i]
        assert len(id_keys[row_id]) == value_counts[values]

    # Verify all keys are accounted for
    all_keys = set(raw_data["key"])
    assert all(key in all_keys for keys in entity_keys.values() for key in keys)
    assert all(key in all_keys for keys in id_keys.values() for key in keys)

    # For empty entities case, verify empty results
    if not selected_entities:
        assert not raw_data["key"]
        assert not entity_keys
        assert not id_keys

    # Verify core variation behavior
    for entity in selected_entities:
        entity_rows = {
            i for i, key in enumerate(raw_data["key"]) if key in entity_keys[entity.id]
        }

        for feature in features:
            values = {raw_data[feature.name][i] for i in entity_rows}
            base_value = entity.base_values[feature.name]

            # Check if base values are included/excluded correctly
            if feature.drop_base:
                assert base_value not in values
            elif not feature.variations:
                assert values == {base_value}

            # Count effective variations (those that actually change the value)
            effective_variations = [
                rule.apply(base_value)
                for rule in feature.variations
                if rule.apply(base_value) != base_value
            ]

            # Check that variations were generated
            if feature.variations:
                expected_count = len(effective_variations) + (
                    0 if feature.drop_base else 1
                )
                assert len(values) == expected_count

    # Verify row count matches expectations with repetition
    for entity in selected_entities:
        # Count effective variations for each feature
        variation_counts = []
        for feature in features:
            base_value = entity.base_values[feature.name]
            effective_variations = [
                rule.apply(base_value)
                for rule in feature.variations
                if rule.apply(base_value) != base_value
            ]
            # Count options: variations + (base if keeping it)
            if feature.drop_base and effective_variations:
                variation_counts.append(len(effective_variations))
            else:
                variation_counts.append(len(effective_variations) + 1)

        # Multiply all counts together to get total combinations, then by repetition
        expected_unique_combinations = functools.reduce(
            lambda x, y: x * y, variation_counts, 1
        )
        expected_total_rows = expected_unique_combinations * (repetition + 1)
        assert len(entity_keys[entity.id]) == expected_total_rows

    # Verify row identity
    # Every id in the data should be tracked in id_keys, and vice versa
    assert set(id_keys.keys()) == set(raw_data["id"])

    # Create a map from values to id
    values_to_id = {}
    for i in range(n_rows):
        values = tuple(raw_data[f.name][i] for f in features)
        row_id = raw_data["id"][i]

        # First encounter of these values, store the id
        if values not in values_to_id:
            values_to_id[values] = row_id
        # If we've seen these values before, make sure the id matches
        else:
            assert values_to_id[values] == row_id

    # Check that different data produces different ids
    if len(unique_values) > 1:
        assert len(set(values_to_id.values())) == len(unique_values)
