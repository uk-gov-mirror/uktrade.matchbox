"""End-to-end tests for the plan API, over a real SQLite warehouse.

Covers the whole plan surface: building a plan with no DAG, laziness, collect with
plan-fingerprint caching, record step storage, lineage navigation, GC, and the terminal
reads (`get_lookup`, `lookup_key`).

Scenario: a dedupe feeding a cross-source link.

    crn: a1=(acme,london) a2=(acme,leeds) a3=(beta,hull)
    dh:  b1=(acme,bristol) b2=(gamma,york)

a1/a2 differ on town, so they are distinct leaves and the deduper does real work.
The apex is a *link*, yet the dedupe grouping of {a1,a2} must survive through it, and
a3/b2, reachable but matched by nothing, must stay singletons (merge-forward).
"""

import json
from collections.abc import Callable

import polars as pl
import pytest
from pydantic import ValidationError
from sqlalchemy import Engine, text

from matchlab import (
    Model,
    Resolver,
    Source,
    Step,
    Transform,
    lineage,
    read_database,
    read_dataframe,
)
from matchlab.adapters import DuckDBAdapter
from matchlab.core.exceptions import StepNotFound
from matchlab.core.kinds import StepKind
from matchlab.models.dedupers import NaiveDeduper
from matchlab.models.linkers import DeterministicLinker
from matchlab.specs import ModelType, SourceSpec
from matchlab.transformers import Explode, Select

# `warehouse`, `adapter` and the `source` factory come from `test/conftest.py`. The
# module-level `_apex`/`_fan_out_plan` builders take that factory, so this file holds no
# source-construction of its own beyond the few tests that need a bespoke shape.


def _dedupe_crn(crn: Source) -> Resolver:
    return crn.dedupe(
        model_class=NaiveDeduper,
        model_settings={"unique_fields": [crn.f("company")]},
    ).resolve()


def _apex(source: Callable[..., Source]) -> tuple[Resolver, Source, Source]:
    """Build the full dedupe → link plan without ever constructing a DAG."""
    crn = source("crn")
    dh = source("dh")
    r_crn = _dedupe_crn(crn)
    apex = r_crn.link(
        dh,
        model_class=DeterministicLinker,
        model_settings={"comparisons": f"l.{crn.f('company')} = r.{dh.f('company')}"},
    ).resolve()
    return apex, crn, dh


def _ids_by_key(matches: pl.DataFrame, column: str) -> dict[str, int]:
    return {
        row[column]: row["root"]
        for row in matches.iter_rows(named=True)
        if row[column] is not None
    }


# -- building and collecting ----------------------------------------------------------


def test_plan_needs_no_dag(source: Callable[..., Source]) -> None:
    """A plan is reachable purely through upstream references, with no DAG object."""
    crn = source("crn")
    assert crn.parents == ()
    assert not crn.is_collected

    deduped = _dedupe_crn(crn)
    assert isinstance(deduped, Resolver)
    # The plan is reachable purely through upstream references.
    assert crn in deduped.lineage()


def test_collect_is_lazy(source: Callable[..., Source]) -> None:
    """Nothing runs until collect. Building a plan touches no warehouse."""
    crn = source("crn")
    deduped = _dedupe_crn(crn)

    assert all(not step.is_collected for step in deduped.lineage())
    deduped.collect()
    assert all(step.is_collected for step in deduped.lineage())


def test_collect_across_sources(source: Callable[..., Source]) -> None:
    """A dedupe feeding a link resolves records across both sources."""
    apex, _crn, _dh = _apex(source)

    lookup = apex.collect().get_lookup()
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


def test_lookup_key_crosses_sources(source: Callable[..., Source]) -> None:
    """lookup_key follows one record's entity across into another source."""
    apex, _crn, _dh = _apex(source)
    apex.collect()

    result = apex.lookup_key(from_source="crn", to_sources=["dh"], key="a1")
    assert set(result["crn"]) == {"a1", "a2"}
    assert set(result["dh"]) == {"b1"}


def test_lookup_key_unknown_raises(source: Callable[..., Source]) -> None:
    """Looking up a key no source holds raises."""
    apex, _crn, _dh = _apex(source)
    apex.collect()
    with pytest.raises(StepNotFound):
        apex.lookup_key(from_source="crn", to_sources=["dh"], key="nope")


# -- laziness and caching -------------------------------------------------------------


def _sabotage(step) -> None:  # noqa: ANN001 - any Step
    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError(f"{step!r} re-ran instead of being read from cache")

    step._execute = boom


def _seal(source) -> None:  # noqa: ANN001 - any Source
    """Make any further warehouse read fail, leaving the memoised one usable."""

    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError(f"{source!r} re-read the warehouse")

    source.fetch = boom


def test_collect_recollect_noop(source: Callable[..., Source]) -> None:
    """Re-collecting an unchanged plan re-runs nothing."""
    apex, _crn, _dh = _apex(source)
    apex.collect()

    for step in apex.lineage():
        _sabotage(step)
    apex.collect()  # must not raise


def test_collect_only_new_steps(
    source: Callable[..., Source],
) -> None:
    """A new downstream branch runs only its new steps, not cached inputs."""
    crn = source("crn")
    deduped = _dedupe_crn(crn)
    deduped.collect()

    for step in deduped.lineage():
        _sabotage(step)

    # A brand-new downstream branch over already-collected inputs.
    dh = source("dh")
    apex = deduped.link(
        dh,
        model_class=DeterministicLinker,
        model_settings={"comparisons": f"l.{crn.f('company')} = r.{dh.f('company')}"},
    ).resolve()
    apex.collect()  # only dh + the new read/model/resolver run

    assert apex.get_lookup().height > 0


def test_source_column_change_invalidates(
    warehouse: Engine, source: Callable[..., Source]
) -> None:
    """Identity is the whole extract, so any selected column moves the fingerprint.

    Without that rule, a column outside identity could change in the warehouse
    without moving the source's fingerprint. The source would cache-hit instead of
    re-storing, and downstream record steps would keep reading the stale value.
    """
    et = "select pk, company, town from crn"

    cleaned = source("crn", et).clean({"town": "crn_town"})
    cleaned.collect()
    assert sorted(cleaned.data()["town"].to_list()) == ["hull", "leeds", "london"]

    with warehouse.begin() as conn:
        conn.execute(text("UPDATE crn SET town = 'oxford' WHERE pk = 'a1'"))

    refreshed = source("crn", et).clean({"town": "crn_town"})
    refreshed.collect()
    assert sorted(refreshed.data()["town"].to_list()) == ["hull", "leeds", "oxford"]


def test_source_identity_is_every_column(source: Callable[..., Source]) -> None:
    """Selecting a column makes it part of the record, so it splits leaves.

    Leave it out of the extract and the rows collapse to one leaf. The SELECT is the
    whole declaration of a source's identity.
    """
    # a1/a2 agree on company and differ on town.
    with_town = source("crn", "select pk, company, town from crn")
    with_town.collect()
    leaves = dict(with_town.leaves().iter_rows())
    assert leaves["a1"] != leaves["a2"]

    without_town = source("crn", "select pk, company from crn")
    without_town.collect()
    leaves = dict(without_town.leaves().iter_rows())
    assert leaves["a1"] == leaves["a2"]


@pytest.mark.parametrize("name", ["crn-x", "crn.x", "1crn", "crn x", ""])
def test_source_name_must_prefix(warehouse: Engine, name: str) -> None:
    """Caught at construction, not as a SQL error three steps later.

    A source's name qualifies every column it contributes, and those land in clean and
    select SQL. `crn-x_company` parses as subtraction, `crn.x_company` as table.column.
    """
    with pytest.raises(ValueError, match="can't prefix a column name"):
        read_database(
            name,
            sql="select pk, company from crn",
            client=warehouse,
            key_field="pk",
        )


def test_source_name_reserved_word(warehouse: Engine) -> None:
    """The name is only ever a prefix, so `select_company` is unambiguous."""
    source = read_database(
        "select",
        sql="select pk, company from crn",
        client=warehouse,
        key_field="pk",
    )
    cleaned = source.clean({"name": f"lower({source.f('company')})"})
    cleaned.collect()
    assert sorted(cleaned.data()["name"].to_list()) == ["acme", "acme", "beta"]


def test_source_key_only_rejected(source: Callable[..., Source]) -> None:
    """A source selecting only its key, nothing to match on, is rejected."""
    source = source("crn", "select pk from crn")
    with pytest.raises(ValueError, match="only its key field"):
        source.collect()


def test_source_no_rows_raises(source: Callable[..., Source]) -> None:
    """A query that returns nothing is a mistake, not an empty source."""
    empty = source("crn", "select pk, company from crn where pk = 'nobody'")
    with pytest.raises(ValueError, match="returned no rows"):
        empty.collect()


def test_source_missing_key_field(warehouse: Engine) -> None:
    """A missing key column is explained, including where the name came from."""
    with pytest.raises(ValueError, match="has no column 'id'") as caught:
        read_database(
            "crn", sql="select pk, company from crn", client=warehouse
        ).collect()
    assert "It returned: pk, company" in str(caught.value)
    assert "defaults to 'id'" in str(caught.value)

    # A frame reaches the same check by the same route.
    with pytest.raises(ValueError, match="has no column 'id'"):
        read_dataframe(
            "dh", df=pl.DataFrame({"pk": ["b1"], "company": ["acme"]})
        ).collect()

    # And so does an explicit key that the location does not return.
    with pytest.raises(ValueError, match="has no column 'pak'"):
        read_database(
            "crn",
            sql="select pk, company from crn",
            client=warehouse,
            key_field="pak",
        ).collect()


def test_source_null_keys_raises(source: Callable[..., Source]) -> None:
    """A null key can't anchor a record, so it's rejected at read time."""
    null_keyed = source("crn", "select null as pk, company from crn")
    with pytest.raises(ValueError, match="has null keys"):
        null_keyed.collect()


def test_source_keys_are_strings(warehouse: Engine) -> None:
    """Whatever the warehouse types the key as, matchlab hands back a string."""
    with warehouse.begin() as conn:
        conn.execute(text("CREATE TABLE nums (id INTEGER, company TEXT)"))
        conn.execute(text("INSERT INTO nums VALUES (1,'acme'),(2,'beta')"))

    source = read_database("nums", sql="select id, company from nums", client=warehouse)
    source.collect()
    assert sorted(source.leaves()["key"].to_list()) == ["1", "2"]


def test_plan_over_dataframes(adapter: DuckDBAdapter) -> None:
    """A whole plan can run with nothing but dataframes."""
    crn = read_dataframe(
        "crn",
        df=pl.DataFrame(
            {
                "pk": ["a1", "a2", "a3"],
                "company": ["acme", "acme", "beta"],
                "town": ["london", "leeds", "hull"],
            }
        ),
        key_field="pk",
    )
    dh = read_dataframe(
        "dh",
        df=pl.DataFrame(
            {"pk": ["b1", "b2"], "company": ["acme", "gamma"], "town": ["a", "b"]}
        ),
        key_field="pk",
    )

    resolved = crn.link(
        dh,
        model_class=DeterministicLinker,
        model_settings={"comparisons": ["l.crn_company = r.dh_company"]},
    ).resolve()
    resolved.collect(adapter)

    lookup = resolved.get_lookup()
    # Every key on both sides is reachable, including the ones nothing matched.
    assert set(lookup["crn_pk"].drop_nulls()) == {"a1", "a2", "a3"}
    assert set(lookup["dh_pk"].drop_nulls()) == {"b1", "b2"}


def test_source_memoises_read(warehouse: Engine, source: Callable[..., Source]) -> None:
    """Re-collecting an existing Source must not re-read the warehouse.

    Asserted against `fetch`, the call that actually goes to the warehouse, rather than
    against `_read_origin`, which is the memo itself. `_ensure` recomputes every
    fingerprint on every collect, so a source's `_spec_key` does run again and does
    re-hash the rows. What it must not do is fetch them again.
    """
    crn = source("crn")
    crn.collect()

    _seal(crn)
    crn.collect()  # the memoised read carries the fingerprint


# -- immutability ---------------------------------------------------------------------


def test_spec_attributes_read_only(source: Callable[..., Source]) -> None:
    """Everything a spec is built from refuses assignment, saying what to do instead."""
    crn = source("crn")
    model = crn.dedupe(
        model_class=NaiveDeduper, model_settings={"unique_fields": ["crn_company"]}
    )
    resolver = model.resolve()

    # Non-comprenehsinve list of read-only attributes
    for step, attribute, value in [
        (crn, "location", None),
        (model, "model_settings", None),
        (resolver, "resolver_settings", None),
    ]:
        with pytest.raises(AttributeError, match="read-only"):
            setattr(step, attribute, value)

    # The settings objects are frozen too, so there is no way round the guard.
    with pytest.raises(ValidationError):
        model.model_instance.unique_fields = ["crn_town"]
    with pytest.raises(ValidationError):
        resolver.resolver_instance.thresholds = {0: 0.9}


def test_parents(source: Callable[..., Source]) -> None:
    """`parents` is derived for all step kinds."""
    crn, dh = source("crn"), source("dh")
    transform = crn.clean({"crn_company": "lower(crn_company)"})
    dedupe = transform.dedupe(NaiveDeduper, {"unique_fields": ["crn_company"]})
    link = crn.link(
        dh,
        model_class=DeterministicLinker,
        model_settings={"comparisons": "l.crn_company = r.dh_company"},
    )
    resolver = dedupe.resolve()

    assert crn.parents == ()  # a source is a leaf
    assert transform.parents == (crn,)
    assert dedupe.parents == (transform,)  # a deduper reads one
    assert link.parents == (crn, dh)  # a linker reads two, left then right
    assert resolver.parents == (dedupe,)  # a resolver reads models

    # Derived, so there is nothing to assign to.
    for step in (crn, transform, dedupe, resolver):
        with pytest.raises(AttributeError):
            step.parents = ()


def test_editable_step_attributes() -> None:
    """The read-only guard does not cover every attribute."""

    class Custom(Step):
        kind = StepKind.SOURCE

        def __init__(self) -> None:
            super().__init__()
            self.label = "mine"

        @property
        def parents(self) -> tuple[Step, ...]:
            return ()

        @property
        def spec(self) -> SourceSpec:  # pragma: no cover - never collected
            raise NotImplementedError

        def _execute(self, adapter: object, fp: bytes) -> None:  # pragma: no cover
            raise NotImplementedError

    assert Custom().label == "mine"


def test_rebuilding_unchanged_cache(source: Callable[..., Source]) -> None:
    """Rebuilding is how a plan changes, and it costs only the steps that moved.

    The source and the transform above the edit address the same artifacts as before,
    so they are cache hits. Only the model and the resolver run again. That is what
    makes rebuilding an acceptable price for immutability.
    """
    crn = source("crn")

    def build(unique_fields: list[str]) -> Resolver:
        return (
            crn.clean({"crn_company": "lower(crn_company)"})
            .dedupe(model_class=NaiveDeduper, model_settings={"unique_fields": unique})
            .resolve()
        )

    unique = ["crn_company"]
    first = build(unique)
    first.collect()
    before = [step._fp for step in first.lineage()]

    unique = ["crn_town"]  # the edit
    second = build(unique)
    second.collect()
    after = [step._fp for step in second.lineage()]

    # source, transform unchanged; model, resolver moved.
    assert before[:2] == after[:2]
    assert before[2:] != after[2:]


def test_derived_class_attributes_drift(source: Callable[..., Source]) -> None:
    """`transformer_class` and `model_type` are read off what they describe."""
    crn = source("crn")
    transform = crn.clean({"crn_company": "lower(crn_company)"})
    model = crn.dedupe(
        model_class=NaiveDeduper, model_settings={"unique_fields": ["crn_company"]}
    )

    assert transform.transformer_class is type(transform.transformer)
    assert transform.spec.transformer_class == "Clean"
    assert model.model_type is ModelType.DEDUPER

    # Neither is a stored attribute, so neither can be written out of agreement.
    with pytest.raises(AttributeError):
        transform.transformer_class = Select
    with pytest.raises(AttributeError):
        model.model_type = ModelType.LINKER


def test_mistyped_settings(source: Callable[..., Source]) -> None:
    """A setting no methodology declares is an error, at every kind of step."""
    crn = source("crn")

    with pytest.raises(ValidationError, match="typo"):
        Source(
            location_class=crn.location_class,
            name="crn",
            location_settings={"sql": "select pk from crn", "typo": 1},
            location_resources=crn.location_resources,
        )

    with pytest.raises(ValidationError, match="max_combinatons"):
        crn.transform(Explode, {"max_combinatons": 5})

    with pytest.raises(ValidationError, match="unique_feilds"):
        crn.dedupe(model_class=NaiveDeduper, model_settings={"unique_feilds": ["x"]})

    deduped = crn.dedupe(
        model_class=NaiveDeduper, model_settings={"unique_fields": [crn.f("company")]}
    )
    with pytest.raises(ValidationError, match="threshold"):
        deduped.resolve(resolver_settings={"threshold": 0.5})


def test_settings_vs_methodology_attr(
    source: Callable[..., Source],
) -> None:
    """`spec` describes the object `_execute` runs, so the two cannot disagree.

    A step used to hold settings passed separately from the instance built out of them,
    which is two things that start equal and can stop being equal. A step now keeps one
    instance and reads its settings back off it.

    That also normalises defaults: a field the call site omitted is still part of the
    configuration, so it belongs in the key. Otherwise the same methodology, written two
    ways, would fingerprint differently and re-run for nothing.
    """
    crn = source("crn")
    model = crn.dedupe(
        model_class=NaiveDeduper, model_settings={"unique_fields": ["crn_company"]}
    )
    assert model.model_settings == model.model_instance.model_dump(mode="json")
    assert model.model_settings["id"] == "id"  # defaulted, never passed

    resolver = model.resolve()
    assert resolver.resolver_settings == resolver.resolver_instance.model_dump(
        mode="json"
    )


# -- Record step storage ---------------------------------------------------------


@pytest.fixture
def computes(monkeypatch: pytest.MonkeyPatch) -> dict[int, int]:
    """Count `Transform._execute` calls per transform, keyed by `id`."""
    counts: dict[int, int] = {}
    original = Transform._execute

    def counting(self: Transform, adapter: DuckDBAdapter, fp: bytes) -> None:
        counts[id(self)] = counts.get(id(self), 0) + 1
        return original(self, adapter, fp)

    monkeypatch.setattr(Transform, "_execute", counting)
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


def _shared_record_step_plan(
    source: Callable[..., Source], comparison: str | None = None
) -> tuple[Resolver, Transform]:
    """One cleaned record step feeding both a dedupe and a link, joined by a resolver.

    `comparison` retunes the linker without touching the record step, so a second plan
    can invalidate the models while the transform's fingerprint still hits.
    """
    crn = source("crn")
    dh = source("dh")
    record_step = crn.clean({"name": "crn_company"})
    deduped = record_step.dedupe(
        model_class=NaiveDeduper, model_settings={"unique_fields": ["name"]}
    )
    linked = record_step.link(
        dh,
        model_class=DeterministicLinker,
        model_settings={"comparisons": comparison or f"l.name = r.{dh.f('company')}"},
    )
    return deduped.resolve(linked), record_step


def test_record_step_stored_with_consumer(
    source: Callable[..., Source], adapter: DuckDBAdapter
) -> None:
    """Collecting a transform's consumer stores the transform too."""
    crn = source("crn")
    record_step = crn.clean({"name": "crn_company"})
    deduped = record_step.dedupe(
        model_class=NaiveDeduper, model_settings={"unique_fields": ["name"]}
    ).resolve()
    deduped.collect()

    assert record_step.is_collected
    assert adapter.has(record_step._fp)


def test_record_step_shared_computed_once(
    source: Callable[..., Source], adapter: DuckDBAdapter, computes: dict[int, int]
) -> None:
    """The point of storing: fan-out costs one computation, not one per consumer."""
    apex, record_step = _shared_record_step_plan(source)

    apex.collect()

    assert computes[id(record_step)] == 1
    assert adapter.has(record_step._fp)


def test_record_step_reused_across_plans(
    source: Callable[..., Source],
    adapter: DuckDBAdapter,
    computes: dict[int, int],
    model_runs: list[Model],
) -> None:
    """A stored record step is read back across sessions, not recomputed.

    The second plan is fresh objects, as a new process would build, with the linker
    retuned so the models genuinely re-run. Their inputs are unchanged, so the transform
    still fingerprints the same and hits cache, reading the record step off disk rather
    than recomputing it.
    """
    dh = source("dh")
    apex, _record_step = _shared_record_step_plan(source)
    apex.collect()
    computes.clear()
    model_runs.clear()

    retuned, record_step = _shared_record_step_plan(
        source, comparison=f"r.{dh.f('company')} = l.name"
    )
    retuned.collect()

    assert computes == {}, "a stored record step was recomputed"
    assert adapter.has(record_step._fp)
    # The linker really did re-run, so something genuinely asked for the record step.
    # Without this the assertion above would hold trivially.
    assert [model.model_class for model in model_runs] == [DeterministicLinker]


def test_record_step_collected_directly(
    source: Callable[..., Source], adapter: DuckDBAdapter
) -> None:
    """A record step collected on its own materialises its table."""
    crn = source("crn")
    record_step = crn.clean({"name": "crn_company"})

    data = record_step.collect().data()
    assert adapter.has(record_step._fp)
    assert "name" in data.columns
    assert data.height == 3


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


def _fan_out_plan(source: Callable[..., Source]) -> tuple[Resolver, list[Model]]:
    """Three models over one shared record step, so the resolver sees repeated readings.

    `Resolver._execute` walks `(model, record_step)` pairs, four of them here. Only two
    distinct readings exist between them, which is what it must collapse to. Linking
    every pair of n sources makes that ratio quadratic.
    """
    crn = source("crn")
    dh = source("dh")
    record_step = crn.clean({"name": "crn_company"})
    models = [
        record_step.dedupe(
            model_class=NaiveDeduper, model_settings={"unique_fields": ["name"]}
        ),
        record_step.link(
            dh,
            model_class=DeterministicLinker,
            model_settings={"comparisons": f"l.name = r.{dh.f('company')}"},
        ),
        record_step.dedupe(
            model_class=NaiveDeduper, model_settings={"unique_fields": ["name", "id"]}
        ),
    ]
    return models[0].resolve(*models[1:]), models


def test_read_identifiers_once_per_source(
    source: Callable[..., Source],
    adapter: DuckDBAdapter,
    identifier_reads: list[tuple[str, bytes | None]],
) -> None:
    """Once per source, not once per model: the quadratic the resolver collapses."""
    apex, models = _fan_out_plan(source)
    assert sum(len(model.parents) for model in apex.parents) == 4  # the pairs it walks

    # Collect the models first so their own reads are done and cleared. What the apex
    # collect records is then the resolver's alone.
    for model in models:
        model.collect()
    identifier_reads.clear()
    apex.collect()

    assert sorted(identifier_reads) == [("crn", None), ("dh", None)]


def test_read_identifiers_dedup_keeps_all(
    source: Callable[..., Source],
) -> None:
    """The merge-forward guarantee, which the dedup must not weaken.

    Every reachable leaf has to reach the resolver output, including records no model
    matched. A reading dropped here loses clusters silently rather than failing, so
    assert on the records rather than on the call count.
    """
    apex, _models = _fan_out_plan(source)

    resolver_output = apex.collect().entities()

    assert dict(resolver_output.group_by("source").len().iter_rows()) == {
        "crn": 3,
        "dh": 2,
    }
    assert set(resolver_output.filter(pl.col("source") == "crn")["key"]) == {
        "a1",
        "a2",
        "a3",
    }
    assert set(resolver_output.filter(pl.col("source") == "dh")["key"]) == {
        "b1",
        "b2",
    }


# -- lineage navigation ---------------------------------------------------------------


def test_lineage_upstream_only(source: Callable[..., Source]) -> None:
    """A step knows its inputs and nothing else, so lineage never looks down."""
    apex, crn, _dh = _apex(source)

    assert crn in apex.lineage()
    assert apex not in crn.lineage()
    assert crn.lineage() == [crn]


def test_lineage_draw_sub_plan(source: Callable[..., Source]) -> None:
    """draw() from a sub-plan shows only that sub-plan."""
    apex, crn, _dh = _apex(source)

    assert "crn" in apex.draw()
    assert "dh" in apex.draw()
    assert "resolver" not in crn.draw()  # a source's sub-plan is just itself


def test_resolver_exposes_sources(source: Callable[..., Source]) -> None:
    """A resolver exposes the sources reachable beneath it."""
    apex, _crn, _dh = _apex(source)
    assert {source.name for source in apex.sources} == {"crn", "dh"}


# -- storage lifetime -----------------------------------------------------------------


def test_gc_keeps_artifacts(
    source: Callable[..., Source], adapter: DuckDBAdapter
) -> None:
    """A store keeps what it was given. Dropping the Python objects reclaims nothing.

    Deliberate. An artifact's value has nothing to do with whether this process still
    holds the variable that produced it. The next process rebuilding the same plan
    wants a cache hit. Reclaiming is the owner's explicit act, not the library's.
    """
    import gc as pygc  # noqa: PLC0415 - the interpreter's, to force reachability

    crn = source("crn")
    deduped = _dedupe_crn(crn)
    deduped.collect()
    fp = deduped._fp
    assert adapter.has(fp)

    del crn, deduped
    pygc.collect()

    assert adapter.has(fp)


def test_fingerprints_name_every_artifact(
    source: Callable[..., Source],
) -> None:
    """Every step in a collected plan carries a fingerprint."""
    crn = source("crn")
    plan = _dedupe_crn(crn)
    plan.collect()

    assert plan.fingerprints() == {step._fp for step in plan.lineage()}


def test_fingerprints_collapse_shared(
    source: Callable[..., Source],
) -> None:
    """Two distinct steps can be the same artifact: same spec, same inputs.

    A set is what makes that safe further down. A store told to keep one of them and
    delete the other would delete the bytes both of them are.
    """
    crn = source("crn")
    cleaning = {"name": "crn_company"}
    twins = [crn.clean(cleaning), crn.clean(cleaning)]
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


def test_fingerprints_uncollected_raises(source: Callable[..., Source]) -> None:
    """An uncollected plan names no artifacts, so it must not answer with a smaller set.

    Silently returning what happens to be collected would tell a caller that less is
    worth keeping than they think. The caller here is about to delete the rest.
    """
    plan = _dedupe_crn(source("crn"))

    with pytest.raises(RuntimeError, match="has not been collected"):
        plan.fingerprints()


def test_prune_to_plan_keeps_cache(
    source: Callable[..., Source], adapter: DuckDBAdapter
) -> None:
    """The property a prune has to have: it removes only what was superseded.

    Editing a cleaning expression strands the whole subtree below it. That is what
    fills a store. Pruning to the plan you kept should take exactly those, and leave a
    store the same plan still hits cache on. If it took one artifact too many the next
    collect quietly recomputes, and the prune has cost work rather than saved space.
    """

    def build(cleaning: str) -> Resolver:
        return (
            source("crn")
            .clean({"name": cleaning})
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

    result = adapter.prune(keep=plan.fingerprints())

    assert result.removed == len(stranded)
    assert all(adapter.has(fp) for fp in live)
    assert not any(adapter.has(fp) for fp in stranded)

    # The real assertion: rebuild the same plan over the pruned store and let it run.
    # Nothing may execute here. If the prune took one artifact too many, this raises.
    rebuilt = build("upper(crn_company)")
    for step in rebuilt.lineage():
        _sabotage(step)
    rebuilt.collect()


# -- reading through a resolver, and grouping -----------------------------------------


def test_resolver_is_a_record_step(source: Callable[..., Source]) -> None:
    """A resolver is a record step, read at `id`=root and matchable on top of directly.

    Reading it returns its records regrouped by entity. Linking it directly to a
    source, with no transform in between, still carries the dedupe grouping forward.
    That's the merge-forward guarantee working through a resolver record step.
    """
    crn = source("crn")
    dh = source("dh")
    deduped = _dedupe_crn(crn)

    # Read as a record step. crn's records sit at `id`=root, so a1/a2 share one id.
    records = deduped.data()
    assert records.height == 3
    assert records["id"].n_unique() == 2

    # Matched on top of directly, with no `.read()` and no transform between.
    apex = deduped.link(
        dh,
        model_class=DeterministicLinker,
        model_settings={"comparisons": f"l.{crn.f('company')} = r.{dh.f('company')}"},
    ).resolve()
    lookup = apex.collect().get_lookup()
    crn_ids = _ids_by_key(lookup, "crn_pk")
    dh_ids = _ids_by_key(lookup, "dh_pk")

    assert crn_ids["a1"] == crn_ids["a2"] == dh_ids["b1"]  # dedupe survived the link
    assert crn_ids["a3"] != crn_ids["a1"]  # fall-through singleton


def test_resolver_read_repeats_per_record(
    source: Callable[..., Source],
) -> None:
    """Without a group, a read is record-grained even when `id` is an entity."""
    crn = source("crn")
    deduped = _dedupe_crn(crn)
    deduped.collect()

    record_step = deduped.clean({"name": "crn_company"})
    record_step.collect()
    data = record_step.data()

    # a1/a2 deduped to one entity, but each contributes a row.
    assert data.height == 3
    assert data["id"].n_unique() == 2


def test_group_one_row_per_entity(source: Callable[..., Source]) -> None:
    """`group` collapses each id, with the aggregate saying how per column."""
    crn = source("crn")
    deduped = _dedupe_crn(crn)
    deduped.collect()

    record_step = deduped.group(
        {
            "name": "any_value(crn_company)",
            "towns": "list(distinct crn_town)",
        }
    )
    record_step.collect()
    data = record_step.data().sort("name")

    assert data.height == 2
    assert data["name"].to_list() == ["acme", "beta"]
    # The deduped entity keeps both towns rather than silently losing one.
    assert sorted(data["towns"][0]) == ["leeds", "london"]


def test_group_merges_forward(source: Callable[..., Source]) -> None:
    """Grouping changes the record step's grain, never the resolver output's.

    Leaves travel via `identifiers()`, read from the adapter, so collapsing rows in
    the record step cannot lose a record from the resolver output below it.
    """
    crn = source("crn")
    dh = source("dh")
    deduped = _dedupe_crn(crn)

    apex = (
        deduped.group({"name": "any_value(crn_company)"})
        .link(
            dh,
            model_class=DeterministicLinker,
            model_settings={"comparisons": f"l.name = r.{dh.f('company')}"},
        )
        .resolve()
    )
    lookup = apex.collect().get_lookup()
    crn_ids = _ids_by_key(lookup, "crn_pk")
    dh_ids = _ids_by_key(lookup, "dh_pk")

    # Every source record still appears, with the dedupe carried forward.
    assert set(crn_ids) == {"a1", "a2", "a3"}
    assert set(dh_ids) == {"b1", "b2"}
    assert crn_ids["a1"] == crn_ids["a2"] == dh_ids["b1"]
    assert crn_ids["a3"] != crn_ids["a1"]


def test_group_multi_source(
    source: Callable[..., Source],
) -> None:
    """Why `group` exists: several sources under one entity.

    Reading two sources through a resolver concatenates diagonally, so crn rows carry
    null dh columns and vice versa, and a comparison on `l.dh_company` is null on
    every crn row. Grouping puts the entity on a single populated row.
    """
    crn = source("crn")
    dh = source("dh")
    linked = crn.link(
        dh,
        model_class=DeterministicLinker,
        model_settings={"comparisons": f"l.{crn.f('company')} = r.{dh.f('company')}"},
    ).resolve()
    linked.collect()

    ungrouped = linked.clean({"c": "crn_company", "d": "dh_company"})
    ungrouped.collect()
    acme = ungrouped.data().filter(pl.col("c") == "acme")
    # Every crn row for the acme entity has a null dh column.
    assert acme["d"].null_count() == acme.height

    grouped = linked.group(
        {
            "company": "any_value(crn_company)",
            "towns": "list(distinct coalesce(crn_town, dh_town))",
        }
    )
    grouped.collect()
    result = grouped.data().filter(pl.col("company") == "acme")

    # One row, both sources' values present, because any_value skips the nulls.
    assert result.height == 1
    assert set(result["towns"][0]) == {"london", "leeds", "bristol"}


def test_group_without_aggregates_raises(source: Callable[..., Source]) -> None:
    """Grouping with no aggregates to say how columns combine is rejected."""
    crn = source("crn")
    with pytest.raises(ValueError, match="aggregate expressions"):
        crn.group({})


def test_explode_cross_source_combinations(source: Callable[..., Source]) -> None:
    """`Explode` gives the cross product across sources, the case `group` skips.

    `group` collapses this diagonally-concatenated record step to one populated row
    (see `test_group_multi_source`). `Explode` instead gives one row per combination of
    each source's populated values.
    """
    crn = source("crn")
    dh = source("dh")
    linked = crn.link(
        dh,
        model_class=DeterministicLinker,
        model_settings={"comparisons": f"l.{crn.f('company')} = r.{dh.f('company')}"},
    ).resolve()
    linked.collect()

    exploded = linked.transform(Explode())
    exploded.collect()
    acme = exploded.data().filter(pl.col("dh_company") == "acme")

    # crn's two towns cross dh's one, both sources' values present on every row.
    assert acme.height == 2
    assert set(acme["crn_town"].to_list()) == {"london", "leeds"}
    assert acme["dh_town"].unique().to_list() == ["bristol"]


# -- specs ----------------------------------------------------------------------------


def test_spec_serialisable(source: Callable[..., Source]) -> None:
    """Each step kind reports its settings through a model, and it round-trips JSON."""
    apex, _crn, _dh = _apex(source)
    apex.collect()

    kinds = set()
    for step in apex.lineage():
        dumped = step.spec.model_dump(mode="json")
        assert json.loads(json.dumps(dumped)) == dumped
        # The spec is the fingerprint payload, so it must be stable.
        assert step._spec_key() == step._spec_key()
        kinds.add(step.kind)

    assert kinds == {"source", "model", "resolver"}


def test_spec_no_upstream_settings(source: Callable[..., Source]) -> None:
    """Specs describe a step's own settings. Edges live on `upstream`."""
    apex, crn, _dh = _apex(source)

    resolver_spec = apex.spec.model_dump(mode="json")
    assert "sql" not in json.dumps(resolver_spec)
    # No upstream reference at all: inputs arrive as parent fingerprints, and a
    # setting that points at one uses its position.
    assert set(resolver_spec) == {"resolver_class", "resolver_settings"}


def test_read_direct_and_through_resolver_distinct(
    source: Callable[..., Source],
) -> None:
    """Reading a source directly and through a resolver are different record steps.

    Read directly, a source is a leaf record step (`id`=leaf). A resolver is itself a
    record step (`id`=root), so matching on it reads the same records regrouped by
    entity. Deduping again over that resolver output is a genuinely different operation
    from the first dedupe, so the two models get different fingerprints.
    """
    crn = source("crn")
    settings = {"unique_fields": [crn.f("company")]}

    first = crn.dedupe(model_class=NaiveDeduper, model_settings=settings)
    deduped = first.resolve()  # a resolver, and a record step reading crn at id=root
    second = deduped.dedupe(model_class=NaiveDeduper, model_settings=settings)

    assert crn.parents == ()  # a source read directly is a leaf
    assert second.left is deduped  # the model reads the resolver directly, no wrapper

    second.resolve().collect()
    assert first._fp != second._fp  # a second pass is not the first one


# -- identity: positions, and published names -----------------------------------------


def test_steps_use_positions(source: Callable[..., Source]) -> None:
    """Nothing in a plan is named.

    Cleaning one source two ways, or comparing two methodologies over it, needs no
    names and cannot collide.
    """
    crn = source("crn")
    strict = crn.clean({"name": f"upper({crn.f('company')})"})
    loose = crn.clean({"name": f"lower({crn.f('company')})"})
    first = strict.dedupe(NaiveDeduper, {"unique_fields": ["name"]})
    second = loose.dedupe(NaiveDeduper, {"unique_fields": ["name", "id"]})

    apex = first.resolve(second)
    apex.collect()

    assert not hasattr(first, "name")
    positions = lineage.number(apex)
    assert positions[id(first)] != positions[id(second)]
    assert apex.entities().height > 0


def test_draw_keys_the_log(source: Callable[..., Source]) -> None:
    """A log line saying `step 4` has to be findable in the plan's drawing."""
    apex, crn, _dh = _apex(source)
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


def test_publish_points_label(
    source: Callable[..., Source], adapter: DuckDBAdapter
) -> None:
    """Publishing is an act on a result, not a property of the plan."""
    apex, _crn, _dh = _apex(source)
    apex.collect(adapter)

    assert adapter.labels() == []  # collecting publishes nothing
    apex.publish("entities")

    assert adapter.labels() == ["entities"]
    assert adapter.find("entities") == apex._fp


def test_publish_idempotent(
    source: Callable[..., Source], adapter: DuckDBAdapter
) -> None:
    """Re-running an unchanged pipeline must not fail. Repointing must be deliberate."""
    apex, crn, _dh = _apex(source)
    apex.collect(adapter).publish("entities")
    apex.publish("entities")  # same resolver output, same name, so this is a no-op

    other = _dedupe_crn(crn).collect(adapter)
    with pytest.raises(
        ValueError, match="already points at a different resolver output"
    ):
        other.publish("entities")

    other.publish("entities", overwrite=True)
    assert adapter.find("entities") == other._fp


def test_publish_needs_collected(
    source: Callable[..., Source],
) -> None:
    """There is nothing to point a name at until the resolver output exists."""
    apex, _crn, _dh = _apex(source)
    with pytest.raises(RuntimeError, match="has not been collected"):
        apex.publish("entities")


# -- resolver settings that point at inputs -------------------------------------------


def test_threshold_stored_as_position(
    source: Callable[..., Source],
) -> None:
    """You hold the model. The plan works out where it sits.

    Positions rather than names are what let inputs be referred to without a naming
    scheme, and what keep the resolver's fingerprint out of it.
    """
    crn = source("crn")
    dh = source("dh")
    first = crn.dedupe(NaiveDeduper, {"unique_fields": [crn.f("company")]})
    second = dh.dedupe(NaiveDeduper, {"unique_fields": [dh.f("company")]})

    resolver = first.resolve(
        second, resolver_settings={"thresholds": {second: 0.8, first: 0.5}}
    )

    assert resolver.resolver_instance.thresholds == {0: 0.5, 1: 0.8}


def test_threshold_must_name_input(source: Callable[..., Source]) -> None:
    """Caught while the model object is still in hand, not deep inside collect."""
    crn = source("crn")
    settings = {"unique_fields": [crn.f("company")]}
    used = crn.dedupe(NaiveDeduper, settings)
    unused = crn.dedupe(NaiveDeduper, settings)

    with pytest.raises(ValueError, match="a model this resolver does not read"):
        used.resolve(resolver_settings={"thresholds": {unused: 0.8}})


def test_edges_keyed_by_position(
    source: Callable[..., Source],
) -> None:
    """The translated thresholds are only useful if the edges arrive aligned to them."""
    crn = source("crn")
    dh = source("dh")
    first = crn.dedupe(NaiveDeduper, {"unique_fields": [crn.f("company")]})
    second = dh.dedupe(NaiveDeduper, {"unique_fields": [dh.f("company")]})

    resolver = first.resolve(second, resolver_settings={"thresholds": {second: 0.8}})

    seen: dict[int, int] = {}

    class Spy:
        """Wrap the methodology rather than patching it.

        `ResolverMethod` is a pydantic model and rejects undeclared attributes.
        """

        def __init__(self, wrapped: object) -> None:
            self.wrapped = wrapped

        def compute_clusters(
            self, model_edges: dict[int, pl.DataFrame]
        ) -> pl.DataFrame:
            seen.update({position: len(df) for position, df in model_edges.items()})
            return self.wrapped.compute_clusters(model_edges=model_edges)

    # Past the read-only guard: a test double, not a configuration change.
    methodology = resolver.resolver_instance
    object.__setattr__(resolver, "resolver_instance", Spy(methodology))
    resolver.collect()

    assert set(seen) == {0, 1}
    assert methodology.thresholds == {1: 0.8}
    assert seen[1] == len(second.edges())
