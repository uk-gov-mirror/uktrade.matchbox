"""Ground-truth tests: does a collected plan recover the entities we planted?

The testkit generates sources whose true entities are known. A perfect matcher then
matches on row *values*, so the edges it emits reference whatever content-derived IDs
the plan actually produced. That makes it possible to assert the resolver output
against truth rather than against hand-built fixtures.

Every test here is the same three steps: generate and write, build a plan, diff it.
`linked` planted the entities, so `.dedupe()`/`.link()` and `.diff_resolver_output()`
all know the answer without being told it. The diff reports *how* a mismatch failed
(perfect/subset/superset/wrong/invalid) rather than only that it did.
"""

import pytest
from sqlalchemy import Engine, create_engine

from matchlab.testkit.linked import LinkedSources, linked_sources_factory

# The `store` fixture comes from `test/conftest.py`.


@pytest.fixture
def warehouse() -> Engine:
    """An empty warehouse: the testkit plants its own tables, so start with none.

    Overrides the shared `crn`/`dh` scenario, which would only get in the way here.
    """
    return create_engine("sqlite:///:memory:")


def planted(n: int, warehouse: Engine) -> LinkedSources:
    """Generate `n` true entities and write them where the plan can read them.

    No `client=` needed: the factory was handed the engine, so the sources already point
    at it.
    """
    return linked_sources_factory(
        n_true_entities=n, engine=warehouse
    ).write_to_location()


def test_dedupe_recovers_entities(warehouse: Engine) -> None:
    """A perfect deduper must resolve each source's records to its true entities."""
    linked = planted(10, warehouse)

    resolver_output = linked.dedupe("crn").model.resolve().collect().entities()

    identical, report = linked.diff_resolver_output(resolver_output, "crn")
    assert identical, report


def test_link_recovers_entities(warehouse: Engine) -> None:
    """A perfect linker must join one entity's records across two sources."""
    linked = planted(10, warehouse)

    resolver_output = linked.link("crn", "cdms").model.resolve().collect().entities()

    # Clusters span both sources and match the planted entities exactly.
    assert set(resolver_output["source"].unique().to_list()) == {"crn", "cdms"}
    identical, report = linked.diff_resolver_output(resolver_output, "crn", "cdms")
    assert identical, report


def test_resolver_output_complete(warehouse: Engine) -> None:
    """No record may be dropped. The resolver output is complete over the source."""
    linked = planted(6, warehouse)

    resolver_output = linked.dedupe("crn").model.resolve().collect().entities()

    expected_keys = {
        str(key) for entity in linked.true_entities for key in entity.get_keys("crn")
    }
    assert set(resolver_output["key"].to_list()) == expected_keys


def test_link_through_dedupe_merges_forward(warehouse: Engine) -> None:
    """A link built *through* an upstream resolver must preserve its grouping.

    This is merge-forward end to end. The apex is a link, but the dedupe's clusters
    have to survive it, and records the link never touched must not collapse. `through=`
    is the whole point of the parameter. It reads the left side as the dedupe resolved
    it, not raw.
    """
    linked = planted(8, warehouse)

    dedupe = linked.dedupe("crn").model.resolve()
    apex = linked.link("crn", "cdms", through=dedupe).model.resolve()
    resolver_output = apex.collect().entities()

    assert set(resolver_output["source"].unique().to_list()) == {"crn", "cdms"}
    identical, report = linked.diff_resolver_output(resolver_output, "crn", "cdms")
    assert identical, report


def test_clusters_keep_entities_distinct(warehouse: Engine) -> None:
    """Precision: no cluster may contain records from two different true entities."""
    linked = planted(12, warehouse)

    resolver_output = linked.dedupe("crn").model.resolve().collect().entities()

    key_to_entity = {
        str(key): entity.id
        for entity in linked.true_entities
        for key in entity.get_keys("crn")
    }
    for row in resolver_output.group_by("root").agg("key").iter_rows(named=True):
        entities = {key_to_entity[key] for key in row["key"]}
        assert len(entities) == 1, f"cluster merged entities {entities}"


def test_diff_reports_failure_mode(warehouse: Engine) -> None:
    """`diff_resolver_output` must classify a bad resolver output, not merely reject it.

    It derives both sides from the testkit, so a bug that compared something against
    itself would make every other test in this file pass. Over-merging must be reported
    as a *superset* of the entities it swallowed.
    """
    linked = planted(8, warehouse)
    resolver_output = linked.link("crn", "cdms").model.resolve().collect().entities()

    # Collapse every record into a single cluster.
    over_merged = resolver_output.with_columns(root=resolver_output["root"].first())

    identical, report = linked.diff_resolver_output(over_merged, "crn", "cdms")
    assert not identical
    assert report["superset"] > 0
    assert report["perfect"] == 0
