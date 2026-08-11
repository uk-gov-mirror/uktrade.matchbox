"""Tests for generating linked sources that share entities."""

import pytest
from sqlalchemy import Engine

from matchlab.testkit.features import FeatureConfig, SourceParameters, SuffixRule
from matchlab.testkit.linked import linked_sources_factory


def test_linked_sources_factory_default() -> None:
    """The default factory builds crn, dh, and cdms, referencing real entities."""
    linked = linked_sources_factory()

    assert "crn" in linked.sources
    assert "dh" in linked.sources
    assert "cdms" in linked.sources

    assert len(linked.true_entities) == 10

    for entity in linked.true_entities:
        source_references = list(entity.keys.items())
        assert len(source_references) > 0

        for source_name, keys in source_references:
            assert source_name in linked.sources
            assert len(keys) > 0


def test_linked_sources_custom_config(sqlite_in_memory_warehouse: Engine) -> None:
    """linked_sources_factory honours per-source feature and entity-count overrides."""
    features = {
        "name": FeatureConfig(
            name="name",
            base_generator="name",
            variations=[SuffixRule(suffix=" Jr")],
        ),
        "user_id": FeatureConfig(
            name="user_id",
            base_generator="uuid4",
        ),
    }

    configs = (
        SourceParameters(
            name="source_a",
            engine=sqlite_in_memory_warehouse,
            features=(features["name"], features["user_id"]),
            n_true_entities=5,
            repetition=1,
        ),
        SourceParameters(
            name="source_b",
            features=(features["name"],),
            n_true_entities=3,
            repetition=2,
        ),
    )

    linked = linked_sources_factory(source_parameters=configs, seed=42)

    # Verify sources were created correctly
    assert set(linked.sources.keys()) == {"source_a", "source_b"}
    assert len(linked.true_entities) == 5  # Max entities from configs

    # Check source A entities
    source_a_entities = [e for e in linked.true_entities if "source_a" in e.keys]
    assert len(source_a_entities) == 5

    # Check source B entities
    source_b_entities = [e for e in linked.true_entities if "source_b" in e.keys]
    assert len(source_b_entities) == 3


def test_linked_sources_find_entities() -> None:
    """find_entities filters true entities by their per-source appearance counts."""
    linked = linked_sources_factory(n_true_entities=10)

    # Find entities that appear at least once in each source
    min_appearances = {"crn": 1, "dh": 1, "cdms": 1}
    common_entities = linked.find_entities(min_appearances=min_appearances)

    # Should be subset of total entities
    assert len(common_entities) <= len(linked.true_entities)

    for entity in common_entities:
        for source, min_count in min_appearances.items():
            assert len(entity.get_keys(source)) >= min_count

    max_appearances = {"dh": 1}
    limited_entities = linked.find_entities(max_appearances=max_appearances)

    for entity in limited_entities:
        for source, max_count in max_appearances.items():
            assert len(entity.get_keys(source)) <= max_count

    # Combined criteria
    filtered_entities = linked.find_entities(
        min_appearances={"crn": 1}, max_appearances={"dh": 2}
    )

    for entity in filtered_entities:
        assert len(entity.get_keys("crn")) >= 1
        assert len(entity.get_keys("dh")) <= 2


def test_entity_value_consistency() -> None:
    """An entity's base value appears in every source row it's referenced from."""
    linked = linked_sources_factory(n_true_entities=5)

    for entity in linked.true_entities:
        base_values = entity.base_values

        for source_name, keys in entity.keys.items():
            source = linked.sources[source_name]
            df = source.data.to_pandas()
            entity_rows = df[df["key"].isin(keys)]

            for feature in source.features:
                if feature.name in base_values and not feature.drop_base:
                    # Skipped when drop_base has removed the base value from the output
                    assert base_values[feature.name] in entity_rows[feature.name].values


def test_source_entity_equality() -> None:
    """A TrueEntity equals a copy of itself, including under hashing."""
    linked = linked_sources_factory(n_true_entities=3)

    entities = list(linked.true_entities)

    assert entities[0] == entities[0]
    assert entities[0] != entities[1]

    entity_copy = entities[0].model_copy()
    assert entity_copy == entities[0]

    entity_set = {entities[0], entity_copy, entities[1]}
    assert len(entity_set) == 2  # entities[0] and its copy hash and compare equal


def test_seed_reproducibility() -> None:
    """The same seed reproduces identical sources. A different seed changes them."""
    source_parameters = SourceParameters(
        name="test_source",
        features=(
            FeatureConfig(
                name="name",
                base_generator="name",
                variations=[SuffixRule(suffix=" Jr")],
            ),
        ),
        n_true_entities=5,
    )

    linked1 = linked_sources_factory(source_parameters=(source_parameters,), seed=42)
    linked2 = linked_sources_factory(source_parameters=(source_parameters,), seed=42)
    linked3 = linked_sources_factory(source_parameters=(source_parameters,), seed=43)

    # Same seed should produce identical results
    assert linked1.sources["test_source"].data.equals(
        linked2.sources["test_source"].data
    )
    assert len(linked1.true_entities) == len(linked2.true_entities)

    # Different seeds should produce different results
    assert not linked1.sources["test_source"].data.equals(
        linked3.sources["test_source"].data
    )


def test_empty_source_handling() -> None:
    """A source configured with zero entities still exists, holding no rows."""
    source_parameters = SourceParameters(
        name="empty_source",
        features=(FeatureConfig(name="name", base_generator="name"),),
        n_true_entities=0,
    )

    linked = linked_sources_factory(source_parameters=(source_parameters,))

    assert "empty_source" in linked.sources
    assert len(linked.sources["empty_source"].data) == 0
    assert len(linked.true_entities) == 0


def test_large_entity_count() -> None:
    """A source can generate a large number of entities without special handling."""
    source_parameters = SourceParameters(
        name="large_source",
        features=(FeatureConfig(name="user_id", base_generator="uuid4"),),
        n_true_entities=10_000,
    )

    linked = linked_sources_factory(source_parameters=(source_parameters,))

    assert len(linked.true_entities) == 10_000
    assert len(linked.sources["large_source"].data) == 10_000


def test_feature_inheritance() -> None:
    """An entity's base values include only the features of the sources it's in."""
    features = {
        "name": FeatureConfig(name="name", base_generator="name"),
        "email": FeatureConfig(name="email", base_generator="email"),
        "phone": FeatureConfig(name="phone", base_generator="phone_number"),
    }

    configs = (
        SourceParameters(
            name="source_a", features=(features["name"], features["email"])
        ),
        SourceParameters(
            name="source_b", features=(features["name"], features["phone"])
        ),
    )

    linked = linked_sources_factory(source_parameters=configs, n_true_entities=10)

    # Check that entities have all relevant features
    for entity in linked.true_entities:
        # All entities should have name (common feature)
        assert "name" in entity.base_values

        # Entities in source_a should have email
        if "source_a" in entity.keys:
            assert "email" in entity.base_values

        # Entities in source_b should have phone
        if "source_b" in entity.keys:
            assert "phone" in entity.base_values


def test_unique_feature_values() -> None:
    """A unique feature never repeats a value, unlike an ordinary one."""
    source_parameters = SourceParameters(
        name="test_source",
        features=(
            FeatureConfig(name="unique_id", base_generator="uuid4", unique=True),
            FeatureConfig(name="is_true", base_generator="boolean", unique=False),
        ),
        n_true_entities=100,
    )

    linked = linked_sources_factory(source_parameters=(source_parameters,))

    unique_ids = set()
    categories = set()
    for entity in linked.true_entities:
        unique_ids.add(entity.base_values["unique_id"])
        categories.add(entity.base_values["is_true"])

    assert len(unique_ids) == 100
    assert len(categories) < 100


def test_source_references() -> None:
    """add_source_reference sets a source's keys and overwrites them next call."""
    linked = linked_sources_factory(n_true_entities=2)
    entity = next(iter(linked.true_entities))

    new_keys = {"keys1", "keys2"}
    entity.add_source_reference("new_source", new_keys)
    assert entity.get_keys("new_source") == new_keys

    updated_keys = {"keys3"}
    entity.add_source_reference("new_source", updated_keys)
    assert entity.get_keys("new_source") == updated_keys

    assert entity.get_keys("nonexistent") == set()


def test_linked_sources_entity_hierarchy() -> None:
    """Every source cluster is a subset of the keys of at least one true entity."""
    features = {
        "name": FeatureConfig(
            name="name",
            base_generator="name",
            variations=[SuffixRule(suffix=" Jr")],
        ),
        "user_id": FeatureConfig(
            name="user_id",
            base_generator="uuid4",
        ),
    }

    configs = (
        SourceParameters(
            name="source_a",
            features=(features["name"], features["user_id"]),
            n_true_entities=5,
        ),
        SourceParameters(
            name="source_b",
            features=(features["name"],),
            n_true_entities=3,
        ),
    )

    linked = linked_sources_factory(source_parameters=configs, seed=42)

    for source_name, source in linked.sources.items():
        for cluster_entity in source.input_clusters:
            matching_parents = [
                true_entity
                for true_entity in linked.true_entities
                if cluster_entity.is_subset_of_true_entity(true_entity)
            ]

            assert len(matching_parents) > 0, (
                f"Cluster in {source_name} has no parent in true_entities"
            )

            assert any(
                cluster_entity.keys <= true_entity.keys
                for true_entity in matching_parents
            ), f"Cluster in {source_name} not a proper subset of any true entity"


def test_linked_sources_entity_count(
    sqlite_in_memory_warehouse: Engine,
) -> None:
    """n_true_entities can be set per source, or overridden for every source at once."""
    base_feature = FeatureConfig(name="name", base_generator="name")

    # A source config that omits n_true_entities is an error, not a silent default
    configs_missing_counts = (
        SourceParameters(
            name="source_a",
            engine=sqlite_in_memory_warehouse,
            features=(base_feature,),
            n_true_entities=5,
        ),
        SourceParameters(
            name="source_b",
            features=(base_feature,),  # Deliberately missing n_true_entities
        ),
    )

    with pytest.raises(
        ValueError, match="n_true_entities not set for sources: source_b"
    ):
        linked_sources_factory(source_parameters=configs_missing_counts)

    # Each source keeps its own requested count
    configs_different_counts = (
        SourceParameters(
            name="source_a",
            engine=sqlite_in_memory_warehouse,
            features=(base_feature,),
            n_true_entities=5,
        ),
        SourceParameters(
            name="source_b",
            features=(base_feature,),
            n_true_entities=10,
        ),
    )

    linked = linked_sources_factory(source_parameters=configs_different_counts)

    # true_entities covers the largest count requested by any one source
    assert len(linked.true_entities) == 10

    source_a_entities = [e for e in linked.true_entities if "source_a" in e.keys]
    source_b_entities = [e for e in linked.true_entities if "source_b" in e.keys]
    assert len(source_a_entities) == 5, "Source A should have 5 entities"
    assert len(source_b_entities) == 10, "Source B should have 10 entities"

    # An explicit factory-level n_true_entities overrides every source's own count
    with pytest.warns(UserWarning, match="factory parameter will be used"):
        linked_override = linked_sources_factory(
            source_parameters=configs_different_counts, n_true_entities=15
        )

    override_source_a = [
        e for e in linked_override.true_entities if "source_a" in e.keys
    ]
    override_source_b = [
        e for e in linked_override.true_entities if "source_b" in e.keys
    ]
    assert len(override_source_a) == 15, "Source A should be overridden to 15 entities"
    assert len(override_source_b) == 15, "Source B should be overridden to 15 entities"
