"""Ground-truth tests: does a collected plan recover the entities we planted?

The testkit generates sources whose true entities are known, then a scripted model
matches on row *values* — so the edges it emits reference whatever content-derived IDs
the plan actually produced. That makes it possible to assert the resolution against
truth rather than against hand-built fixtures.

This is the capability the brief calls out as a differentiator (ER evaluation), and
the reason the testkit was worth porting rather than deleting.
"""

from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, create_engine

from matchlab.adapters import DuckDBAdapter
from matchlab.core.factories.models import model_factory
from matchlab.core.factories.scenarios import link_scenario
from matchlab.core.factories.sources import linked_sources_factory
from matchlab.steps import set_default_adapter


@pytest.fixture(autouse=True)
def adapter() -> Iterator[DuckDBAdapter]:
    store = DuckDBAdapter(":memory:")
    set_default_adapter(store)
    yield store
    set_default_adapter(None)
    store.close()


@pytest.fixture
def warehouse() -> Engine:
    return create_engine("sqlite:///:memory:")


def _partition_by_cluster(resolution) -> set[frozenset[tuple[str, str]]]:  # noqa: ANN001
    """Group (source, key) records by the cluster they resolved to."""
    grouped = resolution.group_by("root").agg("source", "key")
    return {
        frozenset(zip(row["source"], row["key"], strict=True))
        for row in grouped.iter_rows(named=True)
    }


def _partition_from_truth(linked, sources: set[str]) -> set[frozenset[tuple[str, str]]]:  # noqa: ANN001
    """Group (source, key) records by the true entity that owns them."""
    partition = set()
    for entity in linked.true_entities:
        members = {
            (source, str(key)) for source in sources for key in entity.get_keys(source)
        }
        if members:
            partition.add(frozenset(members))
    return partition


def test_dedupe_recovers_the_true_entities(warehouse: Engine) -> None:
    """A perfect deduper must resolve each source's records to its true entities."""
    linked = linked_sources_factory(n_true_entities=10, engine=warehouse)
    for testkit in linked.sources.values():
        testkit.write_to_location(client=warehouse)

    testkit = model_factory(
        left_testkit=linked.sources["crn"],
        true_entities=tuple(linked.true_entities),
    )
    resolution = testkit.resolve().collect().resolution()

    assert _partition_by_cluster(resolution) == _partition_from_truth(linked, {"crn"})


def test_link_recovers_the_true_entities(warehouse: Engine) -> None:
    """A perfect linker must join one entity's records across two sources."""
    linked = linked_sources_factory(n_true_entities=10, engine=warehouse)
    for tk in linked.sources.values():
        tk.write_to_location(client=warehouse)

    testkit = model_factory(
        left_testkit=linked.sources["crn"],
        right_testkit=linked.sources["cdms"],
        true_entities=tuple(linked.true_entities),
    )
    resolution = testkit.resolve().collect().resolution()

    # Clusters span both sources and match the planted entities exactly.
    assert set(resolution["source"].unique().to_list()) == {"crn", "cdms"}
    assert _partition_by_cluster(resolution) == _partition_from_truth(
        linked, {"crn", "cdms"}
    )


def test_resolution_covers_every_record(warehouse: Engine) -> None:
    """No record may be dropped: the resolution is complete over the source."""
    linked = linked_sources_factory(n_true_entities=6, engine=warehouse)
    for tk in linked.sources.values():
        tk.write_to_location(client=warehouse)

    testkit = model_factory(
        left_testkit=linked.sources["crn"],
        true_entities=tuple(linked.true_entities),
    )
    resolution = testkit.resolve().collect().resolution()

    expected_keys = {
        str(key) for entity in linked.true_entities for key in entity.get_keys("crn")
    }
    assert set(resolution["key"].to_list()) == expected_keys


def test_layered_scenario_carries_the_dedupe_forward() -> None:
    """A link built *through* an upstream resolver must preserve its grouping.

    This is merge-forward end to end: the apex is a link, but the dedupe's clusters
    have to survive it, and records the link never touched must not collapse.
    """
    scenario = link_scenario(n_true_entities=8)
    resolution = scenario.apex.collect().resolution()

    assert set(resolution["source"].unique().to_list()) == {"crn", "cdms"}
    assert _partition_by_cluster(resolution) == _partition_from_truth(
        scenario.linked, {"crn", "cdms"}
    )


def test_clusters_never_merge_distinct_entities(warehouse: Engine) -> None:
    """Precision: no cluster may contain records from two different true entities."""
    linked = linked_sources_factory(n_true_entities=12, engine=warehouse)
    for tk in linked.sources.values():
        tk.write_to_location(client=warehouse)

    testkit = model_factory(
        left_testkit=linked.sources["crn"],
        true_entities=tuple(linked.true_entities),
    )
    resolution = testkit.resolve().collect().resolution()

    key_to_entity = {
        str(key): entity.id
        for entity in linked.true_entities
        for key in entity.get_keys("crn")
    }
    for row in resolution.group_by("root").agg("key").iter_rows(named=True):
        entities = {key_to_entity[key] for key in row["key"]}
        assert len(entities) == 1, f"cluster merged entities {entities}"
