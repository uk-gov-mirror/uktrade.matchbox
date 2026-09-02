"""The reads a collected resolver offers, over a real SQLite warehouse.

`entities()` is the whole answer, flat. `get_lookup`, `leaf_sets` and `view_entity` are
projections of it, and none of them rename the resolver output's columns.

Scenario: a dedupe feeding a cross-source link.

    crn (pk, company, town):    a1=(acme,london) a2=(acme,leeds) a3=(beta,hull)
    dh  (pk, company, region):  b1=(acme,south)  b2=(gamma,north)

a1/a2 differ on town, so they are distinct leaves and the deduper does real work. The
sources share `company` but not `town`/`region`, which is what `merge_fields` has to
get right. Three entities result: {a1,a2,b1}, {a3}, {b2}.
"""

from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest
from sqlalchemy import Engine, create_engine, text

from matchlab import Resolver, Source
from matchlab.core.exceptions import StepNotFound
from matchlab.models.dedupers import NaiveDeduper
from matchlab.models.linkers import DeterministicLinker
from matchlab.stores import DuckDBStore

# The `store` fixture comes from `test/conftest.py`.


@pytest.fixture
def warehouse(tmp_path: Path) -> Engine:
    """Overrides the shared `crn`/`dh` scenario. Here `dh` carries `region`, not `town`.

    A second column each source *does not* share is what `view_entity(merge_fields=)`
    has to get right. `company` collapses onto one column, while `town` and `region`
    survive separately, null where the other source has no value.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'wh.sqlite'}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE crn (pk TEXT, company TEXT, town TEXT)"))
        conn.execute(
            text(
                "INSERT INTO crn VALUES "
                "('a1','acme','london'),('a2','acme','leeds'),('a3','beta','hull')"
            )
        )
        conn.execute(text("CREATE TABLE dh (pk TEXT, company TEXT, region TEXT)"))
        conn.execute(
            text("INSERT INTO dh VALUES ('b1','acme','south'),('b2','gamma','north')")
        )
    return engine


def _sources(source: Callable[..., Source]) -> tuple[Source, Source]:
    return source("crn"), source("dh", "select pk, company, region from dh")


@pytest.fixture
def apex(source: Callable[..., Source]) -> Resolver:
    """The collected dedupe → link apex: three entities over crn and dh."""
    crn, dh = _sources(source)
    deduped = crn.dedupe(
        model_class=NaiveDeduper,
        model_settings={"unique_fields": [crn.f("company")]},
    ).resolve()
    return (
        deduped.link(
            dh,
            model_class=DeterministicLinker,
            model_settings={
                "comparisons": f"l.{crn.f('company')} = r.{dh.f('company')}"
            },
        )
        .resolve()
        .collect()
    )


def _root_of(apex: Resolver, source: str, key: str) -> int:
    rows = apex.entities().filter((pl.col("source") == source) & (pl.col("key") == key))
    return rows["root"][0]


# -- entities -------------------------------------------------------------------------


def test_entities_flat(apex: Resolver) -> None:
    """entities() is the whole answer, flat: one row per record, all reachable."""
    resolver_output = apex.entities()

    assert set(resolver_output.columns) == {"root", "leaf", "key", "source"}
    # One row per source record, and every record is reachable.
    assert resolver_output.height == 5
    assert set(resolver_output["key"]) == {"a1", "a2", "a3", "b1", "b2"}

    # The dedupe survived the link, and the link joined acme across sources.
    assert _root_of(apex, "crn", "a1") == _root_of(apex, "crn", "a2")
    assert _root_of(apex, "dh", "b1") == _root_of(apex, "crn", "a1")
    # Fall-through: records no model matched stay in their own entities.
    assert _root_of(apex, "crn", "a3") != _root_of(apex, "crn", "a1")
    assert _root_of(apex, "dh", "b2") != _root_of(apex, "crn", "a1")


def test_entities_collects_first(source: Callable[..., Source]) -> None:
    """A read on an uncollected resolver runs the plan rather than complaining."""
    crn, _dh = _sources(source)
    deduped = crn.dedupe(
        model_class=NaiveDeduper,
        model_settings={"unique_fields": [crn.f("company")]},
    ).resolve()

    assert not deduped.is_collected
    assert deduped.entities().height == 3
    assert deduped.is_collected


def test_entities_filtered(apex: Resolver) -> None:
    """entities(sources) returns only the named sources' rows."""
    assert set(apex.entities(["crn"])["source"]) == {"crn"}


def test_entities_unknown_source_raises(
    apex: Resolver,
) -> None:
    """Silently returning nothing is how a typo stays a typo."""
    with pytest.raises(StepNotFound, match="'nope'"):
        apex.entities(["nope"])

    with pytest.raises(StepNotFound, match="No compatible source"):
        apex.entities([])


# -- get_lookup -----------------------------------------------------------------------


def test_get_lookup_joins_sources(apex: Resolver) -> None:
    """get_lookup joins keys across sources on their shared root."""
    lookup = apex.get_lookup()

    assert set(lookup.columns) == {"root", "crn_pk", "dh_pk"}

    by_key = {
        (row["crn_pk"], row["dh_pk"]): row["root"]
        for row in lookup.iter_rows(named=True)
    }
    # The acme entity holds two crn records and one dh record, so the full join
    # explodes it into a row per pair.
    assert ("a1", "b1") in by_key
    assert ("a2", "b1") in by_key
    # A record with no counterpart keeps its row, with a null on the other side.
    assert ("a3", None) in by_key
    assert (None, "b2") in by_key
    assert by_key[("a1", "b1")] == by_key[("a2", "b1")]


def test_get_lookup_one_source(apex: Resolver) -> None:
    """get_lookup of one source is one row per record, no cross-join."""
    lookup = apex.get_lookup(["crn"])

    assert set(lookup.columns) == {"root", "crn_pk"}
    assert lookup.height == 3
    assert set(lookup["crn_pk"]) == {"a1", "a2", "a3"}


# -- leaf_sets ------------------------------------------------------------------------


def test_leaf_sets(apex: Resolver) -> None:
    """Structure alone, so two resolver outputs of the same records can be compared."""
    sets = apex.leaf_sets()

    assert sorted(len(group) for group in sets) == [1, 1, 3]
    # Deduplicated: a leaf appears once, however many keys point at it.
    assert all(len(group) == len(set(group)) for group in sets)


# -- view_entity ----------------------------------------------------------------------


def test_view_entity_shows_records(apex: Resolver) -> None:
    """view_entity shows the records grouped under one root, by source."""
    acme = apex.view_entity(_root_of(apex, "crn", "a1"))

    # Keys lead, so you can see which source each row came from. Which key leads is
    # lineage order, not something a reader should be asked to predict.
    assert set(acme.columns[:2]) == {"crn_pk", "dh_pk"}
    assert set(acme.columns) == {
        "crn_pk",
        "crn_company",
        "crn_town",
        "dh_pk",
        "dh_company",
        "dh_region",
    }
    assert acme.height == 3
    assert set(acme["crn_pk"].drop_nulls()) == {"a1", "a2"}
    assert set(acme["dh_pk"].drop_nulls()) == {"b1"}
    assert set(acme["crn_town"].drop_nulls()) == {"london", "leeds"}


def test_view_entity_merges_fields(apex: Resolver) -> None:
    """merge_fields collapses the shared column and keeps the rest."""
    acme = apex.view_entity(_root_of(apex, "crn", "a1"), merge_fields=True)

    # `company` exists in both sources and collapses onto one column. `town` and
    # `region` exist in one each and stay, null where the other source has no value.
    assert set(acme.columns) == {"crn_pk", "dh_pk", "company", "town", "region"}
    assert set(acme["company"]) == {"acme"}
    assert acme["town"].null_count() == 1
    assert acme["region"].null_count() == 2
    # Keys are never merged. That is what says where a row came from.
    assert set(acme["crn_pk"].drop_nulls()) == {"a1", "a2"}


def test_view_entity_single_source(apex: Resolver) -> None:
    """A source with no record in the entity contributes no columns."""
    beta = apex.view_entity(_root_of(apex, "crn", "a3"))

    assert set(beta.columns) == {"crn_pk", "crn_company", "crn_town"}
    assert beta.height == 1
    assert beta["crn_pk"][0] == "a3"


def test_view_entity_unknown_raises(apex: Resolver) -> None:
    """view_entity of an unknown root raises."""
    with pytest.raises(KeyError, match="Entity 0 not available"):
        apex.view_entity(0)


def test_view_entity_reads_store(
    warehouse: Engine, source: Callable[..., Source], tmp_path: Path
) -> None:
    """The values shown are the ones the matching saw, and need no connection.

    Reading from the store instead of the warehouse is what keeps these values from
    silently disagreeing with what the matching actually saw.
    """
    store = DuckDBStore(tmp_path / "run.duckdb")
    crn, _dh = _sources(source)
    resolver = crn.dedupe(
        model_class=NaiveDeduper,
        model_settings={"unique_fields": [crn.f("company")]},
    ).resolve()
    resolver.collect(store)

    root = resolver.entities().filter(pl.col("key") == "a1")["root"][0]

    # Not just "don't use the warehouse". Make it impossible to.
    warehouse.dispose()
    (tmp_path / "wh.sqlite").unlink()

    acme = resolver.view_entity(root)
    assert set(acme["crn_pk"]) == {"a1", "a2"}
    assert set(acme["crn_town"]) == {"london", "leeds"}
    store.close()
