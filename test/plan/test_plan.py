"""End-to-end tests for the plan API, over a real SQLite warehouse.

Covers the whole Phase A surface: building a plan with no DAG, laziness, collect with
plan-fingerprint caching, `Clean` fusion, lineage navigation, GC, and the terminal
reads (`get_matches`, `lookup_key`).

Scenario — a dedupe feeding a cross-source link:

    crn: a1=(acme,london) a2=(acme,leeds) a3=(beta,hull)
    dh:  b1=(acme,bristol) b2=(gamma,york)

a1/a2 differ on town, so they are distinct leaves and the deduper does real work.
The apex is a *link*, yet the dedupe grouping of {a1,a2} must survive through it, and
a3/b2 — reachable but matched by nothing — must stay singletons (merge-forward).
"""

import json
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest
from sqlalchemy import Engine, create_engine, text

from matchlab import Resolver, Source, gc, lineage, set_default_adapter
from matchlab.adapters import DuckDBAdapter
from matchlab.core.exceptions import StepNotFound
from matchlab.locations import RelationalDBLocation
from matchlab.models.dedupers import NaiveDeduper
from matchlab.models.linkers import DeterministicLinker


@pytest.fixture
def warehouse(tmp_path: Path) -> Engine:
    engine = create_engine(f"sqlite:///{tmp_path / 'wh.sqlite'}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE crn (pk TEXT, company TEXT, town TEXT)"))
        conn.execute(
            text(
                "INSERT INTO crn VALUES "
                "('a1','acme','london'),('a2','acme','leeds'),('a3','beta','hull')"
            )
        )
        conn.execute(text("CREATE TABLE dh (pk TEXT, company TEXT, town TEXT)"))
        conn.execute(
            text("INSERT INTO dh VALUES ('b1','acme','bristol'),('b2','gamma','york')")
        )
    return engine


@pytest.fixture(autouse=True)
def adapter() -> Iterator[DuckDBAdapter]:
    """Isolate every test behind its own in-memory store."""
    store = DuckDBAdapter(":memory:")
    set_default_adapter(store)
    yield store
    set_default_adapter(None)
    store.close()


def _source(warehouse: Engine, name: str) -> Source:
    location = RelationalDBLocation(name="warehouse")
    location.set_client(warehouse)
    return Source(
        location=location,
        name=name,
        extract_transform=f"select pk, company, town from {name}",
        key_field="pk",
    )


def _dedupe_crn(crn: Source) -> Resolver:
    return crn.dedupe(
        model_class=NaiveDeduper,
        model_settings={"unique_fields": [crn.f("company")]},
    ).resolve()


def _apex(warehouse: Engine) -> tuple[Resolver, Source, Source]:
    """Build the full dedupe → link plan without ever constructing a DAG."""
    crn = _source(warehouse, "crn")
    dh = _source(warehouse, "dh")
    r_crn = _dedupe_crn(crn)
    apex = (
        r_crn.view(crn)
        .link(
            dh,
            model_class=DeterministicLinker,
            model_settings={
                "comparisons": f"l.{crn.f('company')} = r.{dh.f('company')}"
            },
        )
        .resolve()
    )
    return apex, crn, dh


def _ids_by_key(matches: pl.DataFrame, column: str) -> dict[str, int]:
    return {
        row[column]: row["id"]
        for row in matches.iter_rows(named=True)
        if row[column] is not None
    }


# -- building and collecting ----------------------------------------------------------


def test_a_plan_needs_no_dag(warehouse: Engine) -> None:
    crn = _source(warehouse, "crn")
    assert crn.upstream == ()
    assert not crn.is_collected

    deduped = _dedupe_crn(crn)
    assert isinstance(deduped, Resolver)
    # The plan is reachable purely through upstream references.
    assert crn in deduped.lineage()


def test_nothing_runs_until_collect(warehouse: Engine) -> None:
    crn = _source(warehouse, "crn")
    deduped = _dedupe_crn(crn)

    assert all(not step.is_collected for step in deduped.lineage())
    deduped.collect()
    # Clean is fused, so it carries a fingerprint but stores nothing.
    assert all(step.is_collected for step in deduped.lineage())


def test_collect_resolves_across_sources(warehouse: Engine) -> None:
    apex, _crn, _dh = _apex(warehouse)

    lookup = apex.collect().get_matches().as_lookup()
    crn_ids = _ids_by_key(lookup, "crn_pk")
    dh_ids = _ids_by_key(lookup, "dh_pk")

    # Dedupe survived forward through the apex link.
    assert crn_ids["a1"] == crn_ids["a2"]
    # The link joined acme across sources.
    assert dh_ids["b1"] == crn_ids["a1"]
    # Fall-through: records no model matched stay in their own clusters.
    assert crn_ids["a3"] != crn_ids["a1"]
    assert dh_ids["b2"] != crn_ids["a1"]
    assert crn_ids["a3"] != dh_ids["b2"]


def test_lookup_key_crosses_sources(warehouse: Engine) -> None:
    apex, _crn, _dh = _apex(warehouse)
    apex.collect()

    result = apex.lookup_key(from_source="crn", to_sources=["dh"], key="a1")
    assert set(result["crn"]) == {"a1", "a2"}
    assert set(result["dh"]) == {"b1"}


def test_lookup_key_rejects_an_unknown_key(warehouse: Engine) -> None:
    apex, _crn, _dh = _apex(warehouse)
    apex.collect()
    with pytest.raises(StepNotFound):
        apex.lookup_key(from_source="crn", to_sources=["dh"], key="nope")


# -- laziness and caching -------------------------------------------------------------


def _sabotage(step) -> None:  # noqa: ANN001 - any Step
    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError(f"{step.name} re-ran instead of being read from cache")

    step._execute = boom


def test_recollect_runs_nothing(warehouse: Engine) -> None:
    apex, _crn, _dh = _apex(warehouse)
    apex.collect()

    for step in apex.lineage():
        _sabotage(step)
    apex.collect()  # must not raise


def test_building_downstream_only_runs_the_new_steps(warehouse: Engine) -> None:
    crn = _source(warehouse, "crn")
    deduped = _dedupe_crn(crn)
    deduped.collect()

    for step in deduped.lineage():
        _sabotage(step)

    # A brand-new downstream branch over already-collected inputs.
    dh = _source(warehouse, "dh")
    apex = (
        deduped.view(crn)
        .link(
            dh,
            model_class=DeterministicLinker,
            model_settings={
                "comparisons": f"l.{crn.f('company')} = r.{dh.f('company')}"
            },
        )
        .resolve()
    )
    apex.collect()  # only dh + the new clean/model/resolver run

    assert apex.get_matches().as_lookup().height > 0


def _crn_source(warehouse: Engine, extract_transform: str) -> Source:
    location = RelationalDBLocation(name="warehouse")
    location.set_client(warehouse)
    return Source(
        location=location,
        name="crn",
        extract_transform=extract_transform,
        key_field="pk",
    )


def test_a_change_to_any_selected_column_invalidates_the_source(
    warehouse: Engine,
) -> None:
    """Identity is the whole extract, so any selected column moves the fingerprint.

    There is no narrower list of indexed fields for a column to fall outside of, which
    is what used to let a change go unnoticed: the source cache-hit, never re-stored,
    and downstream views kept serving the old value.
    """
    et = "select pk, company, town from crn"

    view = _crn_source(warehouse, et).view(cleaning={"town": "crn_town"})
    view.collect()
    assert sorted(view.data()["town"].to_list()) == ["hull", "leeds", "london"]

    with warehouse.begin() as conn:
        conn.execute(text("UPDATE crn SET town = 'oxford' WHERE pk = 'a1'"))

    refreshed = _crn_source(warehouse, et).view(cleaning={"town": "crn_town"})
    refreshed.collect()
    assert sorted(refreshed.data()["town"].to_list()) == ["hull", "leeds", "oxford"]


def test_identity_is_every_selected_column(warehouse: Engine) -> None:
    """Selecting a column makes it part of the record, so it splits leaves.

    Leave it out of the extract and the rows collapse to one leaf. This is the knob
    that replaced `index_fields`: the SELECT is the whole declaration.
    """
    # a1/a2 agree on company and differ on town.
    with_town = _crn_source(warehouse, "select pk, company, town from crn")
    with_town.collect()
    leaves = dict(with_town.leaves().iter_rows())
    assert leaves["a1"] != leaves["a2"]

    without_town = _crn_source(warehouse, "select pk, company from crn")
    without_town.collect()
    leaves = dict(without_town.leaves().iter_rows())
    assert leaves["a1"] == leaves["a2"]


@pytest.mark.parametrize("name", ["crn-x", "crn.x", "1crn", "crn x", ""])
def test_a_source_name_must_be_able_to_prefix_a_column(
    warehouse: Engine, name: str
) -> None:
    """Caught at construction, not as a SQL error three steps later.

    A source's name qualifies every column it contributes, and those land in cleaning
    SQL. `crn-x_company` parses as subtraction, `crn.x_company` as table.column.
    """
    location = RelationalDBLocation(name="warehouse")
    location.set_client(warehouse)
    with pytest.raises(ValueError, match="can't prefix a column name"):
        Source(
            location=location,
            name=name,
            extract_transform="select pk, company from crn",
            key_field="pk",
        )


def test_a_reserved_word_is_a_fine_source_name(warehouse: Engine) -> None:
    """The name is only ever a prefix, so `select_company` is unambiguous."""
    location = RelationalDBLocation(name="warehouse")
    location.set_client(warehouse)
    source = Source(
        location=location,
        name="select",
        extract_transform="select pk, company from crn",
        key_field="pk",
    )
    view = source.view(cleaning={"name": f"lower({source.f('company')})"})
    view.collect()
    assert sorted(view.data()["name"].to_list()) == ["acme", "acme", "beta"]


def test_a_key_only_extract_is_rejected(warehouse: Engine) -> None:
    source = _crn_source(warehouse, "select pk from crn")
    with pytest.raises(ValueError, match="only its key field"):
        source.collect()


def test_keys_are_read_as_strings(warehouse: Engine) -> None:
    """Whatever the warehouse types the key as, matchlab hands back a string."""
    with warehouse.begin() as conn:
        conn.execute(text("CREATE TABLE nums (id INTEGER, company TEXT)"))
        conn.execute(text("INSERT INTO nums VALUES (1,'acme'),(2,'beta')"))

    location = RelationalDBLocation(name="warehouse")
    location.set_client(warehouse)
    source = Source(
        location=location,
        name="nums",
        extract_transform="select id, company from nums",
        key_field="id",
    )
    source.collect()
    assert sorted(source.leaves()["key"].to_list()) == ["1", "2"]


def test_a_source_memoises_its_read(warehouse: Engine) -> None:
    """Re-collecting an existing Source must not re-read the warehouse."""
    crn = _source(warehouse, "crn")
    crn.collect()

    def boom() -> None:
        raise AssertionError("the warehouse was re-read")

    crn._read_warehouse = boom
    crn.collect()  # memoised fingerprint short-circuits


# -- Clean fusion ---------------------------------------------------------------------


def test_clean_is_fused_by_default(warehouse: Engine, adapter: DuckDBAdapter) -> None:
    crn = _source(warehouse, "crn")
    view = crn.view(cleaning={"name": "crn_company"})
    deduped = view.dedupe(
        model_class=NaiveDeduper, model_settings={"unique_fields": ["name"]}
    ).resolve()
    deduped.collect()

    # It has an identity in the plan, but no artifact was written.
    assert view.is_collected
    assert not adapter.has(view._fp)


def test_collecting_a_clean_directly_materialises_it(
    warehouse: Engine, adapter: DuckDBAdapter
) -> None:
    crn = _source(warehouse, "crn")
    view = crn.view(cleaning={"name": "crn_company"})

    frame = view.collect().data()
    assert adapter.has(view._fp)
    assert set(frame.columns) == {"id", "name"}
    assert frame.height == 3


# -- lineage navigation ---------------------------------------------------------------


def test_lineage_reaches_upstream_only(warehouse: Engine) -> None:
    """A step knows its inputs and nothing else, so lineage never looks down."""
    apex, crn, _dh = _apex(warehouse)

    assert crn in apex.lineage()
    assert apex not in crn.lineage()
    assert crn.lineage() == [crn]


def test_draw_shows_only_the_sub_plan(warehouse: Engine) -> None:
    apex, crn, _dh = _apex(warehouse)

    assert "crn" in apex.draw()
    assert "dh" in apex.draw()
    assert apex.name not in crn.draw()


def test_resolver_exposes_its_sources(warehouse: Engine) -> None:
    apex, _crn, _dh = _apex(warehouse)
    assert {source.name for source in apex.sources} == {"crn", "dh"}


# -- garbage collection ---------------------------------------------------------------


def test_gc_keeps_live_plans(warehouse: Engine) -> None:
    apex, _crn, _dh = _apex(warehouse)
    apex.collect()
    assert gc() == 0  # everything is still referenced


def test_gc_reclaims_a_dropped_plan(warehouse: Engine, adapter: DuckDBAdapter) -> None:
    """Dropping the last reference to a plan makes its storage reclaimable.

    This is what the old DAG registry made impossible: it strong-referenced every
    step forever, so nothing was ever unreachable.
    """
    import gc as pygc  # noqa: PLC0415 - local, to avoid clashing with matchlab.gc

    crn = _source(warehouse, "crn")
    deduped = _dedupe_crn(crn)
    deduped.collect()
    doomed = deduped._fp
    assert adapter.has(doomed)

    del crn, deduped
    pygc.collect()

    assert gc() > 0
    assert not adapter.has(doomed)


# -- views ----------------------------------------------------------------------------


def test_reading_through_a_resolver_repeats_an_entity_per_record(
    warehouse: Engine,
) -> None:
    """Without `group`, a view is record-grained even when `id` is an entity."""
    crn = _source(warehouse, "crn")
    deduped = _dedupe_crn(crn)
    deduped.collect()

    view = deduped.view(crn, cleaning={"name": "crn_company"})
    view.collect()
    frame = view.data()

    # a1/a2 deduped to one entity, but each contributes a row.
    assert frame.height == 3
    assert frame["id"].n_unique() == 2


def test_group_gives_one_row_per_entity(warehouse: Engine) -> None:
    """`group=True` collapses each id, with the aggregate saying how per column."""
    crn = _source(warehouse, "crn")
    deduped = _dedupe_crn(crn)
    deduped.collect()

    view = deduped.view(
        crn,
        cleaning={
            "name": "any_value(crn_company)",
            "towns": "list(distinct crn_town)",
        },
        group=True,
    )
    view.collect()
    frame = view.data().sort("name")

    assert frame.height == 2
    assert frame["name"].to_list() == ["acme", "beta"]
    # The deduped entity keeps both towns rather than silently losing one.
    assert sorted(frame["towns"][0]) == ["leeds", "london"]


def test_a_grouped_view_still_merges_forward(warehouse: Engine) -> None:
    """Grouping changes the view's grain, never the resolution's.

    Leaves travel via `identifiers()`, read from the adapter, so collapsing rows in
    the view cannot lose a record from the resolution below it.
    """
    crn = _source(warehouse, "crn")
    dh = _source(warehouse, "dh")
    deduped = _dedupe_crn(crn)

    apex = (
        deduped.view(crn, cleaning={"name": "any_value(crn_company)"}, group=True)
        .link(
            dh,
            model_class=DeterministicLinker,
            model_settings={"comparisons": f"l.name = r.{dh.f('company')}"},
        )
        .resolve()
    )
    lookup = apex.collect().get_matches().as_lookup()
    crn_ids = _ids_by_key(lookup, "crn_pk")
    dh_ids = _ids_by_key(lookup, "dh_pk")

    # Every source record still appears, with the dedupe carried forward.
    assert set(crn_ids) == {"a1", "a2", "a3"}
    assert set(dh_ids) == {"b1", "b2"}
    assert crn_ids["a1"] == crn_ids["a2"] == dh_ids["b1"]
    assert crn_ids["a3"] != crn_ids["a1"]


def test_group_collapses_a_multi_source_view_onto_one_row(warehouse: Engine) -> None:
    """The case `group` exists for: several sources under one entity.

    Reading two sources through a resolver concatenates diagonally — crn rows carry
    null dh columns and vice versa — so a comparison on `l.dh_company` is null on
    every crn row. Grouping puts the entity on a single populated row.
    """
    crn = _source(warehouse, "crn")
    dh = _source(warehouse, "dh")
    linked = (
        crn.view()
        .link(
            dh,
            model_class=DeterministicLinker,
            model_settings={
                "comparisons": f"l.{crn.f('company')} = r.{dh.f('company')}"
            },
        )
        .resolve()
    )
    linked.collect()

    ungrouped = linked.view(crn, dh, cleaning={"c": "crn_company", "d": "dh_company"})
    ungrouped.collect()
    acme = ungrouped.data().filter(pl.col("c") == "acme")
    # Every crn row for the acme entity has a null dh column.
    assert acme["d"].null_count() == acme.height

    grouped = linked.view(
        crn,
        dh,
        cleaning={
            "company": "any_value(crn_company)",
            "towns": "list(distinct coalesce(crn_town, dh_town))",
        },
        group=True,
    )
    grouped.collect()
    frame = grouped.data().filter(pl.col("company") == "acme")

    # One row, both sources' values present — any_value skips the nulls.
    assert frame.height == 1
    assert set(frame["towns"][0]) == {"london", "leeds", "bristol"}


def test_group_without_cleaning_is_rejected(warehouse: Engine) -> None:
    crn = _source(warehouse, "crn")
    with pytest.raises(ValueError, match="needs cleaning expressions"):
        crn.view(group=True)


# -- configuration --------------------------------------------------------------------


def test_every_step_has_a_serialisable_config(warehouse: Engine) -> None:
    """Each step kind reports its settings through a model, and it round-trips JSON."""
    apex, _crn, _dh = _apex(warehouse)
    apex.collect()

    kinds = set()
    for step in apex.lineage():
        dumped = step.config.model_dump(mode="json")
        assert json.loads(json.dumps(dumped)) == dumped
        # The config is the fingerprint payload, so it must be stable.
        assert step._config_key() == step._config_key()
        kinds.add(step.kind)

    assert kinds == {"source", "clean", "model", "resolver"}


def test_a_config_carries_no_upstream_settings(warehouse: Engine) -> None:
    """Configs describe a step's own settings; edges live on `upstream`."""
    apex, crn, _dh = _apex(warehouse)

    resolver_config = apex.config.model_dump(mode="json")
    assert "extract_transform" not in json.dumps(resolver_config)
    # Upstream names appear where they are load-bearing — per-model thresholds are
    # keyed by them — but the parents' own settings do not.
    assert set(resolver_config["inputs"]) == {model.name for model in apex.inputs}


def test_a_derived_name_distinguishes_reading_through_a_resolver(
    warehouse: Engine,
) -> None:
    """Reading a source directly and through a resolver are different views.

    Two steps in one plan may not share a name, so the derived name has to say
    which it is — and stay usable as an identifier while doing so.
    """
    crn = _source(warehouse, "crn")
    direct = crn.view()
    through = _dedupe_crn(crn).view(crn)

    assert direct.name != through.name
    assert through.config.sources == ("crn",)
    lineage.validate(through)  # duplicate-name detection must pass
