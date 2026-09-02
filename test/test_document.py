"""Round-trip tests for the plan document.

The property that matters is not that a document *parses*, but that a plan rebuilt
from one is the **same plan**: every step fingerprints identically, so a store already
holding the original's artifacts serves them to the rebuilt plan instead of recomputing.
That is what "transfer a plan to another environment for execution" means in practice.

The warehouse here stands in for the target environment: the document travels, the
engine does not, and `load` is handed a client on the other side.
"""

from collections.abc import Callable
from pathlib import Path

import polars as pl
import pytest
from sqlalchemy import Engine, create_engine, text

from matchlab import (
    PlanDocument,
    Resource,
    Source,
    dump,
    lineage,
    load,
    read_database,
    read_dataframe,
)
from matchlab.core.exceptions import ResourceError
from matchlab.models import Model
from matchlab.models.dedupers import NaiveDeduper
from matchlab.models.linkers import DeterministicLinker
from matchlab.resolvers import Resolver
from matchlab.steps import Step
from matchlab.stores import DuckDBStore
from matchlab.transformers import Transform

# `warehouse`, `store` and `source` come from `test/conftest.py`.


@pytest.fixture
def plan(source: Callable[..., Source]) -> Resolver:
    """A dedupe feeding a cross-source link, with one record step read by two models."""
    crn = source("crn")
    dh = source("dh")

    deduped = crn.clean({"company": crn.f("company")}).dedupe(
        model_class=NaiveDeduper,
        model_settings={"unique_fields": ["company"]},
    )
    resolved = deduped.resolve()

    # One record step, read by both models below. This is the structural sharing the
    # document has to preserve rather than inline twice. `resolved` is itself the
    # record step, crn read at id=root.
    entities = resolved.clean({"company": crn.f("company")})
    raw_dh = dh.clean({"company": dh.f("company")})

    comparison = "l.company = r.company"
    linked = entities.link(
        raw_dh,
        model_class=DeterministicLinker,
        model_settings={"comparisons": comparison},
    )
    also_linked = entities.link(
        raw_dh,
        model_class=DeterministicLinker,
        model_settings={"comparisons": f"{comparison} and l.company != 'nope'"},
    )
    return linked.resolve(
        also_linked,
        resolver_settings={"thresholds": {also_linked: 0.5}},
    )


def _parse(root: Step) -> PlanDocument:
    """Dump a plan and read the JSON back, for tests about what a document says.

    `load` takes the JSON as it stands. Parsing is for looking inside.
    """
    return PlanDocument.model_validate_json(dump(root))


# A valid source spec, for tests about a node's shape rather than its settings.
_SOURCE_SPEC = {
    "name": "crn",
    "key_field": "pk",
    "location_class": "RelationalDB",
    "location_settings": {"sql": "select pk from crn"},
}


# -- the round trip -------------------------------------------------------------------


def test_rebuild_fingerprints_match(plan: Resolver, warehouse: Engine) -> None:
    """The whole point: same plan, same fingerprints, so artifacts transfer."""
    original = plan
    original.collect()

    rebuilt = load(dump(original), resources={"warehouse": warehouse})
    rebuilt.collect()

    before = [step._fp for step in lineage.walk(original)]
    after = [step._fp for step in lineage.walk(rebuilt)]
    assert after == before
    assert all(fingerprint is not None for fingerprint in after)


def test_rebuild_hits_cache(
    plan: Resolver, warehouse: Engine, store: DuckDBStore
) -> None:
    """A transferred plan must find the original's artifacts, not redo the work."""
    plan.collect(store)

    rebuilt = load(dump(plan), resources={"warehouse": warehouse})
    for step in lineage.walk(rebuilt):

        def boom(*_a: object, _step: Step = step, **_k: object) -> None:
            raise AssertionError(f"{_step!r} re-ran instead of hitting cache")

        step._execute = boom

    rebuilt.collect(store)  # must not raise


def test_rebuild_different_warehouse(plan: Resolver, tmp_path: Path) -> None:
    """The flip side of the cache hit: a document carries no data, so data decides.

    A source's fingerprint folds in a content hash of what it actually read, and that
    is derived on load from the *target*. Loading the same plan against different rows
    must therefore re-run rather than serve the original's artifacts.
    """
    other = create_engine(f"sqlite:///{tmp_path / 'other.sqlite'}")
    with other.begin() as conn:
        conn.execute(text("CREATE TABLE crn (pk TEXT, company TEXT, town TEXT)"))
        conn.execute(text("INSERT INTO crn VALUES ('a1','different','paris')"))
        conn.execute(text("CREATE TABLE dh (pk TEXT, company TEXT, town TEXT)"))
        conn.execute(text("INSERT INTO dh VALUES ('b1','different','lyon')"))

    original = plan
    original.collect()

    elsewhere = load(dump(original), resources={"warehouse": other})
    elsewhere.collect()

    assert elsewhere._fp != original._fp
    # Same plan, though: the shape is identical, only the data differs.
    assert [step.kind for step in lineage.walk(elsewhere)] == [
        step.kind for step in lineage.walk(original)
    ]


def test_rebuild_same_answer(plan: Resolver, warehouse: Engine) -> None:
    """A rebuilt plan collects to the same lookup as the original."""
    original = plan
    rebuilt = load(dump(original), resources={"warehouse": warehouse})

    expected = original.collect().get_lookup().sort("root")
    actual = rebuilt.collect().get_lookup().sort("root")
    assert actual.equals(expected)


def test_load_from_bytes(plan: Resolver, warehouse: Engine) -> None:
    """A store hands JSON back as bytes as often as text, so `load` takes both.

    The text half of this is every other round trip in this file.
    """
    plan.collect()
    rebuilt = load(dump(plan).encode(), resources={"warehouse": warehouse})
    rebuilt.collect()

    assert [step._fp for step in lineage.walk(rebuilt)] == [
        step._fp for step in lineage.walk(plan)
    ]


def test_document_carries_no_labels(plan: Resolver, warehouse: Engine) -> None:
    """Publishing is done to a result, so a label is not part of the plan.

    A source's name is not a label: it prefixes that source's columns and tags its
    rows, so it is part of the output rather than a way of finding it.
    """
    document = _parse(plan)

    assert not any(hasattr(node, "name") for node in document.steps)
    sources = [node.spec for node in document.steps if node.kind == "source"]
    assert {spec.name for spec in sources} == {"crn", "dh"}

    rebuilt = load(dump(plan), resources={"warehouse": warehouse})
    assert lineage.number(rebuilt) == {
        id(step): index for index, step in enumerate(lineage.walk(rebuilt))
    }


def test_document_preserves_sharing(plan: Resolver, warehouse: Engine) -> None:
    """A view feeding two models is one node referenced twice, not two nodes."""
    original = plan
    document = _parse(original)

    assert len(document.steps) == len(lineage.walk(original))

    rebuilt = load(dump(original), resources={"warehouse": warehouse})
    models = [step for step in lineage.walk(rebuilt) if isinstance(step, Model)]
    linkers = [model for model in models if model.right is not None]
    assert len(linkers) == 2
    # Both linkers hold the *same* view object, not two equal ones.
    assert linkers[0].left is linkers[1].left


def test_bare_value(warehouse: Engine) -> None:
    """A bare value can be used instead of a resource, until dump is attemped."""
    source = read_database("crn", sql="select pk, company from crn", client=warehouse)

    assert source.sample().height > 0  # usable here

    with pytest.raises(ResourceError, match="unnamed resource for 'client'"):
        dump(source)


def test_document_carries_no_client(plan: Resolver) -> None:
    """Locations describe where data lives; connecting is the target's business."""
    serialised = dump(plan)

    assert "sqlite" not in serialised  # the engine's URL never leaves
    assert '"resources":{"client":"warehouse"}' in serialised  # only its name does

    with pytest.raises(ResourceError, match="'warehouse' — step"):
        load(serialised, resources={})


def test_resource_rename_keeps_fingerprint(plan: Resolver, warehouse: Engine) -> None:
    """Renaming a resource changes no fingerprint anywhere in the plan.

    The guarantee the `*_settings` / `*_resources` split exists to give: a resource is
    named in its own field, so nothing it is called can reach a spec.
    """
    original = plan
    original.collect()

    document = _parse(original)
    elsewhere = document.model_copy(
        update={
            "steps": tuple(
                node.model_copy(
                    update={
                        "resources": dict.fromkeys(node.resources, "somewhere_else")
                    }
                )
                for node in document.steps
            )
        }
    )

    renamed = load(elsewhere.model_dump_json(), resources={"somewhere_else": warehouse})
    renamed.collect()

    assert [step._fp for step in lineage.walk(renamed)] == [
        step._fp for step in lineage.walk(original)
    ]
    sources = [step for step in lineage.walk(renamed) if isinstance(step, Source)]
    assert {source.location_resources["client"].name for source in sources} == {
        "somewhere_else"
    }


def test_required_resources_list(warehouse: Engine) -> None:
    """A caller can see what to supply, before loading or from the error after."""
    frame = pl.DataFrame({"pk": ["b1"], "company": ["acme"]})
    db = read_database(
        "crn", sql="select pk, company from crn", client=Resource("wh", warehouse)
    )
    memory = read_dataframe("dh", df=Resource("dh_frame", frame), key_field="pk")
    plan = db.link(
        memory,
        model_class="DeterministicLinker",
        model_settings={"comparisons": ["l.crn_company = r.dh_company"]},
    ).resolve()

    document = _parse(plan)
    assert document.required_resources() == {"wh", "dh_frame"}

    with pytest.raises(ResourceError, match="'dh_frame' — step"):
        load(dump(plan), resources={"wh": warehouse})

    # Supply none and every missing name comes back at once, so working from the
    # error takes one more attempt rather than one per resource.
    with pytest.raises(ResourceError) as complaint:
        load(dump(plan), resources={})
    assert "'wh'" in str(complaint.value)
    assert "'dh_frame'" in str(complaint.value)


def test_document_custom_location(plan: Resolver, warehouse: Engine) -> None:
    """Documents can represent custom locations."""
    document = _parse(plan)
    broken = document.model_copy(
        update={
            "steps": tuple(
                node.model_copy(
                    update={
                        "spec": node.spec.model_copy(
                            update={"location_class": "S3Location"}
                        )
                    }
                )
                if node.kind == "source"
                else node
                for node in document.steps
            )
        }
    )

    # It parses — a document may legitimately name a class this codebase lacks.
    assert PlanDocument.model_validate_json(broken.model_dump_json())

    with pytest.raises(ValueError, match="No location class named 'S3Location'"):
        load(broken.model_dump_json(), resources={"warehouse": warehouse})


# -- what the document says -----------------------------------------------------------


def test_edges_point_backwards(plan: Resolver) -> None:
    """Topological order is the document's ordering guarantee."""
    document = _parse(plan)

    assert document.steps[0].kind == "source"
    assert document.steps[-1].kind == "resolver"
    for position, node in enumerate(document.steps):
        assert all(target < position for target in node.inputs)


def test_settings_vs_resources(warehouse: Engine) -> None:
    """A node keeps the query in its spec and the client's *name* beside it."""
    source = read_database(
        "ch", sql="select id, c from ch", client=Resource("wh", warehouse)
    )
    node = _parse(source).steps[0]

    assert node.spec.location_settings == {"sql": "select id, c from ch"}
    assert node.resources == {"client": "wh"}


def test_spec_has_settings_not_edges(plan: Resolver) -> None:
    """The split that makes a spec safe to hash and a document able to rebuild."""
    document = _parse(plan)

    for node in document.steps:
        spec = node.spec.model_dump(mode="json")
        if node.kind == "view":
            assert set(spec) == {"cleaning", "group"}
        if node.kind == "model":
            assert set(spec) == {"model_type", "model_class", "model_settings"}
        # A source's own name is settings, not an edge: it prefixes its columns.
        if node.kind == "source":
            assert spec["name"] in {"crn", "dh"}


def test_setting_travels_as_position(plan: Resolver, warehouse: Engine) -> None:
    """Thresholds key by position, which has to survive JSON's string-only keys.

    A name would have been the alternative, and would have made the document depend on
    names being stable — the thing positions exist to avoid.
    """
    document = _parse(plan)
    apex = document.steps[-1]

    assert apex.kind == "resolver"
    # `1` is the second input, and JSON has stringified the key.
    assert apex.spec.resolver_settings == {"thresholds": {"1": 0.5}}

    rebuilt = load(dump(plan), resources={"warehouse": warehouse})
    assert rebuilt.resolver_instance.thresholds == {1: 0.5}


_MODEL_SPEC = {
    "model_type": "deduper",
    "model_class": "NaiveDeduper",
    "model_settings": {"unique_fields": ["company"]},
}


def test_document_rejects_forward_edge() -> None:
    """Inputs before consumers is an invariant, not a convention."""
    with pytest.raises(ValueError, match="not an earlier step"):
        PlanDocument.model_validate(
            {
                "steps": [
                    {"kind": "model", "spec": _MODEL_SPEC, "inputs": [1]},
                    {"kind": "model", "spec": _MODEL_SPEC, "inputs": []},
                ]
            }
        )


def test_spec_parsed_by_kind() -> None:
    """The specs are structurally close, so `kind` discriminates, not a plain union."""
    document = PlanDocument.model_validate(
        {
            "steps": [
                {
                    "kind": "model",
                    "spec": {
                        "model_type": "deduper",
                        "model_class": "NaiveDeduper",
                        "model_settings": {"unique_fields": ["company"]},
                    },
                    "inputs": [],
                }
            ]
        }
    )
    assert document.steps[0].spec.model_class == "NaiveDeduper"


def test_load_rejects_wrong_kind(plan: Resolver, warehouse: Engine) -> None:
    """A malformed document fails at load, not with a confusing error much later."""
    document = _parse(plan)
    resolver = next(node for node in document.steps if node.kind == "resolver")
    broken = PlanDocument(
        steps=tuple(
            node.model_copy(update={"inputs": (0,)}) if node is resolver else node
            for node in document.steps
        )
    )

    with pytest.raises(ValueError, match="must read only"):
        load(broken.model_dump_json(), resources={"warehouse": warehouse})


def test_document_empty_rejected() -> None:
    """A document with no steps is rejected on load."""
    with pytest.raises(ValueError, match="at least one step"):
        load(PlanDocument(steps=()).model_dump_json(), resources={})


def test_rebuild_reads_resolver_as_record_step(
    plan: Resolver, warehouse: Engine
) -> None:
    """A resolver is a record step, so a rebuilt transform reads one at `id`=root.

    The `plan` cleans `crn`'s dedupe resolver, so a rebuilt transform must read a
    `Resolver` as its upstream, the layering shape, preserved across the round trip.
    """
    rebuilt = load(dump(plan), resources={"warehouse": warehouse})
    reading_resolver = [
        step
        for step in lineage.walk(rebuilt)
        if isinstance(step, Transform) and isinstance(step.parents[0], Resolver)
    ]

    assert len(reading_resolver) == 1
    assert [s.name for s in reading_resolver[0].parents[0].sources] == ["crn"]


def test_rebuild_transform_chain(
    source: Callable[..., Source], warehouse: Engine
) -> None:
    """A select → clean chain rebuilds with the transformer settings intact.

    Transformers carry their configuration by name, so a document names the class and
    dumps its fields. The rebuilt transform must reconstruct both, in chain order.
    """
    crn = source("crn")
    plan = (
        crn.select("crn_company", "crn_town")
        .clean({"name": "lower(crn_company)"})
        .dedupe(NaiveDeduper, {"unique_fields": ["name"]})
        .resolve()
    )

    rebuilt = load(dump(plan), resources={"warehouse": warehouse})
    transforms = [step for step in lineage.walk(rebuilt) if isinstance(step, Transform)]

    assert [t.transformer_class.__name__ for t in transforms] == ["Select", "Clean"]
    plan.collect()
    rebuilt.collect(store=DuckDBStore(":memory:"))
    assert plan.fingerprints() == rebuilt.fingerprints()
