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

import pytest
from sqlalchemy import Engine, create_engine, text

from matchlab import PlanDocument, Source, dump, lineage, load
from matchlab.adapters import DuckDBAdapter
from matchlab.frames import Resolved
from matchlab.models import Model
from matchlab.models.dedupers import NaiveDeduper
from matchlab.models.linkers import DeterministicLinker
from matchlab.resolvers import Resolver
from matchlab.steps import Step
from matchlab.transformers import Transform

# `warehouse`, `adapter` and `source` come from `test/conftest.py`.


@pytest.fixture
def plan(source: Callable[..., Source]) -> Resolver:
    """A dedupe feeding a cross-source link, with one frame read by two models."""
    crn = source("crn")
    dh = source("dh")

    deduped = crn.clean({"company": crn.f("company")}).dedupe(
        model_class=NaiveDeduper,
        model_settings={"unique_fields": ["company"]},
    )
    resolved = deduped.resolve()

    # One frame, read by both models below — the structural sharing the document has to
    # preserve rather than inline twice.
    entities = resolved.read(crn).clean({"company": crn.f("company")})
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


def _transfer(document: PlanDocument) -> PlanDocument:
    """Send a document over the wire and back — JSON is the transfer format."""
    return PlanDocument.model_validate_json(document.model_dump_json())


# A valid source spec, for tests about a node's shape rather than its settings.
_SOURCE_SPEC = {
    "name": "crn",
    "extract_transform": "select pk from crn",
    "key_field": "pk",
}

# A valid location reference, likewise.
_LOCATION_REF = {"location_class": "RelationalDBLocation", "name": "warehouse"}


# -- the round trip -------------------------------------------------------------------


def test_rebuild_fingerprints_match(plan: Resolver, warehouse: Engine) -> None:
    """The whole point: same plan, same fingerprints, so artifacts transfer."""
    original = plan
    original.collect()

    rebuilt = load(_transfer(dump(original)), clients={"warehouse": warehouse})
    rebuilt.collect()

    before = [step._fp for step in lineage.walk(original)]
    after = [step._fp for step in lineage.walk(rebuilt)]
    assert after == before
    assert all(fingerprint is not None for fingerprint in after)


def test_rebuild_hits_cache(
    plan: Resolver, warehouse: Engine, adapter: DuckDBAdapter
) -> None:
    """A transferred plan must find the original's artifacts, not redo the work."""
    plan.collect(adapter)

    rebuilt = load(_transfer(dump(plan)), clients={"warehouse": warehouse})
    for step in lineage.walk(rebuilt):

        def boom(*_a: object, _step: Step = step, **_k: object) -> None:
            raise AssertionError(f"{_step!r} re-ran instead of hitting cache")

        step._execute = boom

    rebuilt.collect(adapter)  # must not raise


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

    document = _transfer(dump(original))
    elsewhere = load(document, clients={"warehouse": other})
    elsewhere.collect()

    assert elsewhere._fp != original._fp
    # Same plan, though: the shape is identical, only the data differs.
    assert [step.kind for step in lineage.walk(elsewhere)] == [
        step.kind for step in lineage.walk(original)
    ]


def test_rebuild_same_answer(plan: Resolver, warehouse: Engine) -> None:
    """A rebuilt plan collects to the same lookup as the original."""
    original = plan
    rebuilt = load(_transfer(dump(original)), clients={"warehouse": warehouse})

    expected = original.collect().get_lookup().sort("root")
    actual = rebuilt.collect().get_lookup().sort("root")
    assert actual.equals(expected)


def test_document_carries_no_labels(plan: Resolver, warehouse: Engine) -> None:
    """Publishing is done to a result, so a label is not part of the plan.

    A source's name is not a label: it prefixes that source's columns and tags its
    rows, so it is part of the output rather than a way of finding it.
    """
    document = dump(plan)

    assert not any(hasattr(node, "name") for node in document.steps)
    sources = [node.spec for node in document.steps if node.kind == "source"]
    assert {spec.name for spec in sources} == {"crn", "dh"}

    rebuilt = load(_transfer(document), clients={"warehouse": warehouse})
    assert lineage.number(rebuilt) == {
        id(step): index for index, step in enumerate(lineage.walk(rebuilt))
    }


def test_document_preserves_sharing(plan: Resolver, warehouse: Engine) -> None:
    """A view feeding two models is one node referenced twice, not two nodes."""
    original = plan
    document = dump(original)

    assert len(document.steps) == len(lineage.walk(original))

    rebuilt = load(document, clients={"warehouse": warehouse})
    models = [step for step in lineage.walk(rebuilt) if isinstance(step, Model)]
    linkers = [model for model in models if model.right is not None]
    assert len(linkers) == 2
    # Both linkers hold the *same* view object, not two equal ones.
    assert linkers[0].left is linkers[1].left


def test_document_carries_no_client(plan: Resolver) -> None:
    """Locations describe where data lives; connecting is the target's business."""
    document = dump(plan)
    serialised = document.model_dump_json()

    assert "sqlite" not in serialised  # the engine's URL never leaves
    assert '"name":"warehouse"' in serialised  # but the client it needs is named

    with pytest.raises(ValueError, match="has no client"):
        load(document, clients={})


def test_location_rename_keeps_fingerprint(plan: Resolver, warehouse: Engine) -> None:
    """A location says how to rebuild, so it travels on the node and not in the spec.

    Renaming a warehouse changes no byte any source produces.
    """
    original = plan
    original.collect()

    document = _transfer(dump(original))
    renamed = load(
        document.model_copy(
            update={
                "steps": tuple(
                    node.model_copy(
                        update={
                            "location": node.location.model_copy(
                                update={"name": "somewhere_else"}
                            )
                        }
                    )
                    if node.kind == "source"
                    else node
                    for node in document.steps
                )
            }
        ),
        clients={"somewhere_else": warehouse},
    )
    renamed.collect()

    assert [step._fp for step in lineage.walk(renamed)] == [
        step._fp for step in lineage.walk(original)
    ]
    sources = [step for step in lineage.walk(renamed) if isinstance(step, Source)]
    assert {source.location.name for source in sources} == {"somewhere_else"}


def test_document_custom_location(plan: Resolver, warehouse: Engine) -> None:
    """Documents can represent custom locations."""
    document = dump(plan)
    broken = document.model_copy(
        update={
            "steps": tuple(
                node.model_copy(
                    update={
                        "location": node.location.model_copy(
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
    assert _transfer(broken)

    with pytest.raises(ValueError, match="No location class named 'S3Location'"):
        load(broken, clients={"warehouse": warehouse})


def test_location_source_only() -> None:
    """A location reference binds a client, and only a source needs one bound."""
    with pytest.raises(ValueError, match="must name the location"):
        PlanDocument.model_validate(
            {"steps": [{"kind": "source", "spec": _SOURCE_SPEC, "inputs": []}]}
        )

    with pytest.raises(ValueError, match="resolved step must not name a location"):
        PlanDocument.model_validate(
            {
                "steps": [
                    {
                        "kind": "resolved",
                        "spec": {},
                        "inputs": [],
                        "location": _LOCATION_REF,
                    }
                ]
            }
        )


# -- what the document says -----------------------------------------------------------


def test_edges_point_backwards(plan: Resolver) -> None:
    """Topological order is the document's ordering guarantee."""
    document = dump(plan)

    assert document.steps[0].kind == "source"
    assert document.steps[-1].kind == "resolver"
    for position, node in enumerate(document.steps):
        assert all(target < position for target in node.inputs)


def test_spec_has_settings_not_edges(plan: Resolver) -> None:
    """The split that makes a spec safe to hash and a document able to rebuild."""
    document = dump(plan)

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
    document = _transfer(dump(plan))
    apex = document.steps[-1]

    assert apex.kind == "resolver"
    # `1` is the second input, and JSON has stringified the key.
    assert apex.spec.resolver_settings == {"thresholds": {"1": 0.5}}

    rebuilt = load(document, clients={"warehouse": warehouse})
    assert rebuilt.resolver_settings.thresholds == {1: 0.5}


def test_document_rejects_forward_edge() -> None:
    """Inputs before consumers is an invariant, not a convention."""
    with pytest.raises(ValueError, match="not an earlier step"):
        PlanDocument.model_validate(
            {
                "steps": [
                    {"kind": "resolved", "spec": {}, "inputs": [1]},
                    {"kind": "resolved", "spec": {}, "inputs": []},
                ]
            }
        )


def test_spec_parsed_by_kind() -> None:
    """`ResolvedSpec` has no fields, so a plain union would swallow anything."""
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
    document = dump(plan)
    resolver = next(node for node in document.steps if node.kind == "resolver")
    broken = PlanDocument(
        steps=tuple(
            node.model_copy(update={"inputs": (0,)}) if node is resolver else node
            for node in document.steps
        )
    )

    with pytest.raises(ValueError, match="must read only"):
        load(broken, clients={"warehouse": warehouse})


def test_document_empty_rejected() -> None:
    """A document with no steps is rejected on load."""
    with pytest.raises(ValueError, match="at least one step"):
        load(PlanDocument(steps=()), clients={})


def test_rebuild_resolved_read(plan: Resolver, warehouse: Engine) -> None:
    """A resolved read's inputs are sources plus one resolver, told apart by kind."""
    rebuilt = load(dump(plan), clients={"warehouse": warehouse})
    reads = [step for step in lineage.walk(rebuilt) if isinstance(step, Resolved)]

    assert len(reads) == 1
    assert isinstance(reads[0].resolver, Resolver)
    assert [source.name for source in reads[0].sources] == ["crn"]


def test_rebuild_transform_chain(
    source: Callable[..., Source], warehouse: Engine
) -> None:
    """A select → clean chain rebuilds with the transformer settings intact.

    Transformers carry their configuration by name, so a document names the class and
    dumps its fields; the rebuilt transform must reconstruct both, in chain order.
    """
    crn = source("crn")
    plan = (
        crn.select("crn_company", "crn_town")
        .clean({"name": "lower(crn_company)"})
        .dedupe(NaiveDeduper, {"unique_fields": ["name"]})
        .resolve()
    )

    rebuilt = load(_transfer(dump(plan)), clients={"warehouse": warehouse})
    transforms = [step for step in lineage.walk(rebuilt) if isinstance(step, Transform)]

    assert [t.transformer_class.__name__ for t in transforms] == ["Select", "Clean"]
    plan.collect()
    rebuilt.collect(adapter=DuckDBAdapter(":memory:"))
    assert plan.fingerprints() == rebuilt.fingerprints()
