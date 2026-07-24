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

from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest
from sqlalchemy import Engine, create_engine, text

from matchlab import Resolve, Source, gc, set_default_adapter
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
        index_fields=["company", "town"],
    )


def _dedupe_crn(crn: Source) -> Resolve:
    return crn.dedupe(
        model_class=NaiveDeduper,
        model_settings={"unique_fields": [crn.f("company")]},
    ).resolve()


def _apex(warehouse: Engine) -> tuple[Resolve, Source, Source]:
    """Build the full dedupe → link plan without ever constructing a DAG."""
    crn = _source(warehouse, "crn")
    dh = _source(warehouse, "dh")
    r_crn = _dedupe_crn(crn)
    apex = (
        r_crn.clean(crn)
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
    assert isinstance(deduped, Resolve)
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
        deduped.clean(crn)
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


def test_a_change_to_a_non_indexed_column_invalidates_the_source(
    warehouse: Engine,
) -> None:
    """The fingerprint must cover the whole extract, not just the indexed fields.

    `town` is selected but not indexed, so it never reaches a leaf hash — but `Clean`
    reads it out of the stored extract. Hashing only the index would leave this change
    invisible, the source would cache-hit without re-storing, and the view would keep
    serving the old value.
    """
    location = RelationalDBLocation(name="warehouse")
    location.set_client(warehouse)

    def build() -> Source:
        return Source(
            location=location,
            name="crn",
            extract_transform="select pk, company, town from crn",
            key_field="pk",
            index_fields=["company"],  # town rides along unindexed
        )

    view = build().clean({"town": "crn_town"})
    view.collect()
    assert sorted(view.data()["town"].to_list()) == ["hull", "leeds", "london"]

    with warehouse.begin() as conn:
        conn.execute(text("UPDATE crn SET town = 'oxford' WHERE pk = 'a1'"))

    refreshed = build().clean({"town": "crn_town"})
    refreshed.collect()
    assert sorted(refreshed.data()["town"].to_list()) == ["hull", "leeds", "oxford"]


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
    view = crn.clean({"name": "crn_company"})
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
    view = crn.clean({"name": "crn_company"})

    frame = view.collect().data()
    assert adapter.has(view._fp)
    assert set(frame.columns) == {"id", "name"}
    assert frame.height == 3


# -- lineage navigation ---------------------------------------------------------------


def test_get_step_is_lineage_scoped(warehouse: Engine) -> None:
    apex, crn, _dh = _apex(warehouse)

    assert apex.get_step("crn") is crn
    with pytest.raises(StepNotFound):
        crn.get_step(apex.name)


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
