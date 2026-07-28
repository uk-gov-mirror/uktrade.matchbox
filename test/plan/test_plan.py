"""End-to-end tests for the plan API, over a real SQLite warehouse.

Covers the whole Phase A surface: building a plan with no DAG, laziness, collect with
plan-fingerprint caching, `View` storage, lineage navigation, GC, and the terminal
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

from matchlab import Model, Resolver, Source, View, lineage, set_default_adapter
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
    location = RelationalDBLocation(name="warehouse", client=warehouse)
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
        raise AssertionError(f"{step!r} re-ran instead of being read from cache")

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
    apex.collect()  # only dh + the new view/model/resolver run

    assert apex.get_matches().as_lookup().height > 0


def _crn_source(warehouse: Engine, extract_transform: str) -> Source:
    location = RelationalDBLocation(name="warehouse", client=warehouse)
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
    location = RelationalDBLocation(name="warehouse", client=warehouse)
    with pytest.raises(ValueError, match="can't prefix a column name"):
        Source(
            location=location,
            name=name,
            extract_transform="select pk, company from crn",
            key_field="pk",
        )


def test_a_reserved_word_is_a_fine_source_name(warehouse: Engine) -> None:
    """The name is only ever a prefix, so `select_company` is unambiguous."""
    location = RelationalDBLocation(name="warehouse", client=warehouse)
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

    location = RelationalDBLocation(name="warehouse", client=warehouse)
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


# -- View storage ---------------------------------------------------------------------


@pytest.fixture
def computes(monkeypatch: pytest.MonkeyPatch) -> dict[int, int]:
    """Count `View._compute` calls per view, keyed by `id(view)`."""
    counts: dict[int, int] = {}
    original = View._compute

    def counting(self: View, adapter: DuckDBAdapter) -> pl.DataFrame:
        counts[id(self)] = counts.get(id(self), 0) + 1
        return original(self, adapter)

    monkeypatch.setattr(View, "_compute", counting)
    return counts


@pytest.fixture
def model_runs(monkeypatch: pytest.MonkeyPatch) -> list[Model]:
    """Record which models actually executed, rather than hitting cache."""
    ran: list[Model] = []
    original = Model._execute

    def counting(self: Model, adapter: DuckDBAdapter, fp: bytes) -> None:
        ran.append(self)
        return original(self, adapter, fp)

    monkeypatch.setattr(Model, "_execute", counting)
    return ran


def _shared_view_plan(
    warehouse: Engine, comparison: str | None = None
) -> tuple[Resolver, View]:
    """One cleaned view feeding both a dedupe and a link, joined by a resolver.

    `comparison` retunes the linker without touching the view, so a second plan can
    invalidate the models while every view's fingerprint still hits.
    """
    crn = _source(warehouse, "crn")
    dh = _source(warehouse, "dh")
    view = crn.view(cleaning={"name": "crn_company"})
    deduped = view.dedupe(
        model_class=NaiveDeduper, model_settings={"unique_fields": ["name"]}
    )
    linked = view.link(
        dh,
        model_class=DeterministicLinker,
        model_settings={"comparisons": comparison or f"l.name = r.{dh.f('company')}"},
    )
    return deduped.resolve(linked), view


def test_a_view_is_stored_when_its_consumer_is_collected(
    warehouse: Engine, adapter: DuckDBAdapter
) -> None:
    crn = _source(warehouse, "crn")
    view = crn.view(cleaning={"name": "crn_company"})
    deduped = view.dedupe(
        model_class=NaiveDeduper, model_settings={"unique_fields": ["name"]}
    ).resolve()
    deduped.collect()

    assert view.is_collected
    assert adapter.has(view._fp)


def test_a_shared_view_is_computed_once(
    warehouse: Engine, adapter: DuckDBAdapter, computes: dict[int, int]
) -> None:
    """The point of storing: fan-out costs one computation, not one per consumer."""
    apex, view = _shared_view_plan(warehouse)

    apex.collect()

    assert computes[id(view)] == 1
    assert adapter.has(view._fp)


def test_a_rebuilt_plan_reads_a_view_stored_by_an_earlier_one(
    warehouse: Engine,
    adapter: DuckDBAdapter,
    computes: dict[int, int],
    model_runs: list[Model],
) -> None:
    """Cross-session reuse: a stored view is read back, not recomputed.

    The second plan is fresh objects, as a new process would build, with the linker
    retuned so the models genuinely re-run. Their inputs are unchanged, so the views
    hit cache and the frame comes off disk. Recomputing here was the old behaviour:
    a rebuilt view had no way to know its table was already stored.
    """
    dh = _source(warehouse, "dh")
    apex, _view = _shared_view_plan(warehouse)
    apex.collect()
    computes.clear()
    model_runs.clear()

    retuned, view = _shared_view_plan(
        warehouse, comparison=f"r.{dh.f('company')} = l.name"
    )
    retuned.collect()

    assert computes == {}, "a stored view was recomputed"
    assert adapter.has(view._fp)
    # The linker really did re-run, so something genuinely asked for the frame —
    # without this the assertion above would hold trivially.
    assert [model.model_class for model in model_runs] == [DeterministicLinker]


def test_collecting_a_view_directly_materialises_it(
    warehouse: Engine, adapter: DuckDBAdapter
) -> None:
    crn = _source(warehouse, "crn")
    view = crn.view(cleaning={"name": "crn_company"})

    frame = view.collect().data()
    assert adapter.has(view._fp)
    assert set(frame.columns) == {"id", "name"}
    assert frame.height == 3


# -- identifiers ----------------------------------------------------------------------


@pytest.fixture
def identifier_reads(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, bytes | None]]:
    """Record the `(source_name, resolver_fp)` of every `read_identifiers` call."""
    calls: list[tuple[str, bytes | None]] = []
    original = DuckDBAdapter.read_identifiers

    def counting(
        self: DuckDBAdapter,
        source_fp: bytes,
        source_name: str,
        resolver_fp: bytes | None = None,
    ) -> pl.DataFrame:
        calls.append((source_name, resolver_fp))
        return original(self, source_fp, source_name, resolver_fp)

    monkeypatch.setattr(DuckDBAdapter, "read_identifiers", counting)
    return calls


def _fan_out_plan(warehouse: Engine) -> tuple[Resolver, list[Model]]:
    """Three models over one shared view, so the resolver sees repeated readings.

    `Resolver._execute` walks `(model, view)` pairs — four of them here. Only two
    distinct readings exist between them, which is what it must collapse to. Linking
    every pair of n sources makes that ratio quadratic.
    """
    crn = _source(warehouse, "crn")
    dh = _source(warehouse, "dh")
    view = crn.view(cleaning={"name": "crn_company"})
    models = [
        view.dedupe(
            model_class=NaiveDeduper, model_settings={"unique_fields": ["name"]}
        ),
        view.link(
            dh,
            model_class=DeterministicLinker,
            model_settings={"comparisons": f"l.name = r.{dh.f('company')}"},
        ),
        view.dedupe(
            model_class=NaiveDeduper, model_settings={"unique_fields": ["name", "id"]}
        ),
    ]
    return models[0].resolve(*models[1:]), models


def test_a_resolver_reads_identifiers_once_per_source_not_once_per_model(
    warehouse: Engine,
    adapter: DuckDBAdapter,
    identifier_reads: list[tuple[str, bytes | None]],
) -> None:
    apex, models = _fan_out_plan(warehouse)
    assert sum(len(model.inputs) for model in apex.inputs) == 4  # the pairs it walks

    # Collect the models first so their own reads are done and cleared; what the apex
    # collect records is then the resolver's alone.
    for model in models:
        model.collect()
    identifier_reads.clear()
    apex.collect()

    assert sorted(identifier_reads) == [("crn", None), ("dh", None)]


def test_deduplicating_the_readings_keeps_every_record(warehouse: Engine) -> None:
    """The merge-forward guarantee, which the dedup must not weaken.

    Every reachable leaf has to reach the resolution — including records no model
    matched. A reading dropped here loses clusters silently rather than failing, so
    assert on the records rather than on the call count.
    """
    apex, _models = _fan_out_plan(warehouse)

    resolution = apex.collect().resolution()

    assert dict(resolution.group_by("source").len().iter_rows()) == {"crn": 3, "dh": 2}
    assert set(resolution.filter(pl.col("source") == "crn")["key"]) == {
        "a1",
        "a2",
        "a3",
    }
    assert set(resolution.filter(pl.col("source") == "dh")["key"]) == {"b1", "b2"}


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
    assert "resolver" not in crn.draw()  # a source's sub-plan is just itself


def test_resolver_exposes_its_sources(warehouse: Engine) -> None:
    apex, _crn, _dh = _apex(warehouse)
    assert {source.name for source in apex.sources} == {"crn", "dh"}


# -- storage lifetime -----------------------------------------------------------------


def test_dropping_a_plan_leaves_its_artifacts_stored(
    warehouse: Engine, adapter: DuckDBAdapter
) -> None:
    """A store keeps what it was given; dropping the Python objects reclaims nothing.

    Deliberate. An artifact's value has nothing to do with whether this process still
    holds the variable that produced it — the next process rebuilding the same plan
    wants a cache hit. Reclaiming is the owner's explicit act, not the library's.
    """
    import gc as pygc  # noqa: PLC0415 - the interpreter's, to force reachability

    crn = _source(warehouse, "crn")
    deduped = _dedupe_crn(crn)
    deduped.collect()
    fp = deduped._fp
    assert adapter.has(fp)

    del crn, deduped
    pygc.collect()

    assert adapter.has(fp)


def test_fingerprints_name_every_artifact_a_plan_is_made_of(warehouse: Engine) -> None:
    crn = _source(warehouse, "crn")
    plan = _dedupe_crn(crn)
    plan.collect()

    assert plan.fingerprints() == {step._fp for step in plan.lineage()}


def test_fingerprints_collapse_two_nodes_that_address_one_artifact(
    warehouse: Engine,
) -> None:
    """Two distinct steps can be the same artifact — same spec over same inputs.

    A set is what makes that safe further down: a store told to keep one of them and
    delete the other would delete the bytes both of them are.
    """
    crn = _source(warehouse, "crn")
    cleaning = {"name": "crn_company"}
    twins = [crn.view(cleaning=cleaning), crn.view(cleaning=cleaning)]
    plan = (
        twins[0]
        .dedupe(model_class=NaiveDeduper, model_settings={"unique_fields": ["name"]})
        .resolve(
            twins[1].dedupe(
                model_class=NaiveDeduper, model_settings={"unique_fields": ["name"]}
            )
        )
    )
    plan.collect()

    assert twins[0] is not twins[1]
    assert twins[0]._fp == twins[1]._fp
    assert len(plan.fingerprints()) < len(plan.lineage())


def test_fingerprints_refuse_an_uncollected_plan(warehouse: Engine) -> None:
    """An uncollected plan names no artifacts, so it must not answer with a smaller set.

    Silently returning what happens to be collected would tell a caller that less is
    worth keeping than they think — and the caller here is about to delete the rest.
    """
    plan = _dedupe_crn(_source(warehouse, "crn"))

    with pytest.raises(RuntimeError, match="has not been collected"):
        plan.fingerprints()


def test_trimming_to_a_plan_leaves_it_fully_cached(
    warehouse: Engine, adapter: DuckDBAdapter
) -> None:
    """The property a trim has to have: it removes only what was superseded.

    Editing a cleaning expression strands the whole subtree below it — that is what
    fills a store. Trimming to the plan you kept should take exactly those, and leave a
    store the same plan still hits cache on. If it took one artifact too many the next
    collect quietly recomputes, and the trim has cost work rather than saved space.
    """

    def build(cleaning: str) -> Resolver:
        return (
            _source(warehouse, "crn")
            .view(cleaning={"name": cleaning})
            .dedupe(
                model_class=NaiveDeduper, model_settings={"unique_fields": ["name"]}
            )
            .resolve()
        )

    first = build("crn_company")
    first.collect()
    superseded = {step._fp for step in first.lineage()}

    plan = build("upper(crn_company)")  # the edit
    plan.collect()
    live = {step._fp for step in plan.lineage()}

    stranded = superseded - live
    assert stranded, "the edit should have stranded something"

    result = adapter.trim(keep=plan.fingerprints())

    assert result.removed == len(stranded)
    assert all(adapter.has(fp) for fp in live)
    assert not any(adapter.has(fp) for fp in stranded)

    # The real assertion: rebuild the same plan over the trimmed store and let it run.
    # Nothing may execute — if the trim took one artifact too many, this raises.
    rebuilt = build("upper(crn_company)")
    for step in rebuilt.lineage():
        _sabotage(step)
    rebuilt.collect()


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


# -- specs ----------------------------------------------------------------------------


def test_every_step_has_a_serialisable_spec(warehouse: Engine) -> None:
    """Each step kind reports its settings through a model, and it round-trips JSON."""
    apex, _crn, _dh = _apex(warehouse)
    apex.collect()

    kinds = set()
    for step in apex.lineage():
        dumped = step.spec.model_dump(mode="json")
        assert json.loads(json.dumps(dumped)) == dumped
        # The spec is the fingerprint payload, so it must be stable.
        assert step._spec_key() == step._spec_key()
        kinds.add(step.kind)

    assert kinds == {"source", "view", "model", "resolver"}


def test_a_spec_carries_no_upstream_settings(warehouse: Engine) -> None:
    """Specs describe a step's own settings; edges live on `upstream`."""
    apex, crn, _dh = _apex(warehouse)

    resolver_spec = apex.spec.model_dump(mode="json")
    assert "extract_transform" not in json.dumps(resolver_spec)
    # No upstream reference at all: inputs arrive as parent fingerprints, and a
    # setting that points at one uses its position.
    assert set(resolver_spec) == {"resolver_class", "resolver_settings"}


def test_a_view_through_a_resolver_is_a_different_step(warehouse: Engine) -> None:
    """Reading a source directly and through a resolver are different views.

    The edge is what distinguishes them: it is on `upstream`, and folded into the
    fingerprint in order.
    """
    crn = _source(warehouse, "crn")
    settings = {"unique_fields": [crn.f("company")]}

    first = crn.view().dedupe(model_class=NaiveDeduper, model_settings=settings)
    deduped = first.resolve()
    through = deduped.view(crn)
    second = through.dedupe(model_class=NaiveDeduper, model_settings=settings)

    assert through.upstream == (crn, deduped)
    assert crn.view().upstream == (crn,)

    second.resolve().collect()
    assert first._fp != second._fp  # a second pass is not the first one


# -- identity: positions, and published names -----------------------------------------


def test_steps_have_no_names_only_positions(warehouse: Engine) -> None:
    """Nothing in a plan is named. Cleaning one source two ways, or comparing two
    methodologies over it, needs no names and cannot collide."""
    crn = _source(warehouse, "crn")
    strict = crn.view(cleaning={"name": f"upper({crn.f('company')})"})
    loose = crn.view(cleaning={"name": f"lower({crn.f('company')})"})
    first = strict.dedupe(NaiveDeduper, {"unique_fields": ["name"]})
    second = loose.dedupe(NaiveDeduper, {"unique_fields": ["name", "id"]})

    apex = first.resolve(second)
    apex.collect()

    assert not hasattr(first, "name")
    positions = lineage.number(apex)
    assert positions[id(first)] != positions[id(second)]
    assert apex.resolution().height > 0


def test_a_drawing_is_the_key_to_the_log(warehouse: Engine) -> None:
    """A log line saying `step 4` has to be findable in the plan's drawing."""
    apex, crn, _dh = _apex(warehouse)
    apex.collect()

    drawing = apex.draw()
    for step, position in lineage.number(apex).items():
        del step
        assert f"[{position}] " in drawing
    # A source is drawn with its name, because a source has one.
    assert f"source '{crn.name}'" in drawing
    assert "model(DeterministicLinker)" in drawing
    assert "model(NaiveDeduper)" in drawing
    assert "resolver(Components)" in drawing


def test_publishing_points_a_label_at_a_resolution(
    warehouse: Engine, adapter: DuckDBAdapter
) -> None:
    """Publishing is an act on a result, not a property of the plan."""
    apex, _crn, _dh = _apex(warehouse)
    apex.collect(adapter)

    assert adapter.labels() == []  # collecting publishes nothing
    apex.publish("entities")

    assert adapter.labels() == ["entities"]
    assert adapter.find("entities") == apex._fp


def test_publishing_is_idempotent_but_will_not_silently_move_a_label(
    warehouse: Engine, adapter: DuckDBAdapter
) -> None:
    """Re-running an unchanged pipeline must not fail; repointing must be deliberate."""
    apex, crn, _dh = _apex(warehouse)
    apex.collect(adapter).publish("entities")
    apex.publish("entities")  # same resolution, same name — a no-op

    other = _dedupe_crn(crn).collect(adapter)
    with pytest.raises(ValueError, match="already points at a different resolution"):
        other.publish("entities")

    other.publish("entities", overwrite=True)
    assert adapter.find("entities") == other._fp


def test_publishing_needs_a_collected_resolution(warehouse: Engine) -> None:
    """There is nothing to point a name at until the resolution exists."""
    apex, _crn, _dh = _apex(warehouse)
    with pytest.raises(RuntimeError, match="has not been collected"):
        apex.publish("entities")


# -- resolver settings that point at inputs -------------------------------------------


def test_a_threshold_names_a_model_and_is_stored_as_a_position(
    warehouse: Engine,
) -> None:
    """You hold the model; the plan works out where it sits.

    Positions rather than names are what let inputs be referred to without a naming
    scheme, and what keep the resolver's fingerprint out of it.
    """
    crn = _source(warehouse, "crn")
    dh = _source(warehouse, "dh")
    first = crn.dedupe(NaiveDeduper, {"unique_fields": [crn.f("company")]})
    second = dh.dedupe(NaiveDeduper, {"unique_fields": [dh.f("company")]})

    resolver = first.resolve(
        second, resolver_settings={"thresholds": {second: 0.8, first: 0.5}}
    )

    assert resolver.resolver_settings.thresholds == {0: 0.5, 1: 0.8}


def test_a_threshold_must_name_an_input(warehouse: Engine) -> None:
    """Caught while the model object is still in hand, not deep inside collect."""
    crn = _source(warehouse, "crn")
    settings = {"unique_fields": [crn.f("company")]}
    used = crn.dedupe(NaiveDeduper, settings)
    unused = crn.dedupe(NaiveDeduper, settings)

    with pytest.raises(ValueError, match="a model this resolver does not read"):
        used.resolve(resolver_settings={"thresholds": {unused: 0.8}})


def test_edges_reach_the_methodology_keyed_by_the_same_positions(
    warehouse: Engine,
) -> None:
    """The translated thresholds are only useful if the edges arrive aligned to them."""
    crn = _source(warehouse, "crn")
    dh = _source(warehouse, "dh")
    first = crn.dedupe(NaiveDeduper, {"unique_fields": [crn.f("company")]})
    second = dh.dedupe(NaiveDeduper, {"unique_fields": [dh.f("company")]})

    resolver = first.resolve(second, resolver_settings={"thresholds": {second: 0.8}})

    seen: dict[int, int] = {}

    class Spy:
        """`ResolverMethod` is a pydantic model and rejects undeclared attributes,
        so wrap the methodology rather than patching it."""

        def __init__(self, wrapped: object) -> None:
            self.wrapped = wrapped

        def compute_clusters(
            self, model_edges: dict[int, pl.DataFrame]
        ) -> pl.DataFrame:
            seen.update({position: len(df) for position, df in model_edges.items()})
            return self.wrapped.compute_clusters(model_edges=model_edges)

    resolver.resolver_instance = Spy(resolver.resolver_instance)
    resolver.collect()

    assert set(seen) == {0, 1}
    assert resolver.resolver_settings.thresholds == {1: 0.8}
    assert seen[1] == len(second.edges())
