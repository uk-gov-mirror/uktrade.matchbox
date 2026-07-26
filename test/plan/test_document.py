"""Round-trip tests for the plan document.

The property that matters is not that a document *parses*, but that a plan rebuilt
from one is the **same plan**: every step fingerprints identically, so a store already
holding the original's artifacts serves them to the rebuilt plan instead of recomputing.
That is what "transfer a plan to another environment for execution" means in practice.

The warehouse here stands in for the target environment: the document travels, the
engine does not, and `load` is handed a client on the other side.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine, text

from matchlab import PlanDocument, Source, dump, lineage, load, set_default_adapter
from matchlab.adapters import DuckDBAdapter
from matchlab.locations import RelationalDBLocation
from matchlab.models import Model
from matchlab.models.dedupers import NaiveDeduper
from matchlab.models.linkers import DeterministicLinker
from matchlab.resolvers import Resolver
from matchlab.steps import Step
from matchlab.views import View


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


def _plan(warehouse: Engine) -> Resolver:
    """A dedupe feeding a cross-source link, with one view read by two models."""
    crn = _source(warehouse, "crn")
    dh = _source(warehouse, "dh")

    deduped = crn.view(cleaning={"company": crn.f("company")}).dedupe(
        model_class=NaiveDeduper,
        model_settings={"unique_fields": ["company"]},
    )
    resolved = deduped.resolve()

    # One view, read by both models below — the structural sharing the document has to
    # preserve rather than inline twice.
    entities = resolved.view(crn, cleaning={"company": crn.f("company")})
    raw_dh = dh.view(cleaning={"company": dh.f("company")})

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


# -- the round trip -------------------------------------------------------------------


def test_a_rebuilt_plan_fingerprints_identically(warehouse: Engine) -> None:
    """The whole point: same plan, same fingerprints, so artifacts transfer."""
    original = _plan(warehouse)
    original.collect()

    rebuilt = load(_transfer(dump(original)), clients={"warehouse": warehouse})
    rebuilt.collect()

    before = [step._fp for step in lineage.walk(original)]
    after = [step._fp for step in lineage.walk(rebuilt)]
    assert after == before
    assert all(fingerprint is not None for fingerprint in after)


def test_a_rebuilt_plan_hits_cache_rather_than_recomputing(
    warehouse: Engine, adapter: DuckDBAdapter
) -> None:
    """A transferred plan must find the original's artifacts, not redo the work."""
    _plan(warehouse).collect(adapter)

    rebuilt = load(_transfer(dump(_plan(warehouse))), clients={"warehouse": warehouse})
    for step in lineage.walk(rebuilt):

        def boom(*_a: object, _step: Step = step, **_k: object) -> None:
            raise AssertionError(f"{_step!r} re-ran instead of hitting cache")

        step._execute = boom

    rebuilt.collect(adapter)  # must not raise


def test_a_different_warehouse_gives_different_fingerprints(
    warehouse: Engine, tmp_path: Path
) -> None:
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

    original = _plan(warehouse)
    original.collect()

    document = _transfer(dump(original))
    elsewhere = load(document, clients={"warehouse": other})
    elsewhere.collect()

    assert elsewhere._fp != original._fp
    # Same plan, though: the shape is identical, only the data differs.
    assert [step.kind for step in lineage.walk(elsewhere)] == [
        step.kind for step in lineage.walk(original)
    ]


def test_a_rebuilt_plan_produces_the_same_answer(warehouse: Engine) -> None:
    original = _plan(warehouse)
    rebuilt = load(_transfer(dump(original)), clients={"warehouse": warehouse})

    expected = original.collect().get_matches().as_lookup().sort("id")
    actual = rebuilt.collect().get_matches().as_lookup().sort("id")
    assert actual.equals(expected)


def test_a_document_carries_no_labels_only_source_names(warehouse: Engine) -> None:
    """Publishing is done to a result, so a label is not part of the plan.

    A source's name is not a label: it prefixes that source's columns and tags its
    rows, so it is part of the output rather than a way of finding it.
    """
    document = dump(_plan(warehouse))

    assert not any(hasattr(node, "name") for node in document.steps)
    sources = [node.config for node in document.steps if node.kind == "source"]
    assert {config.name for config in sources} == {"crn", "dh"}

    rebuilt = load(_transfer(document), clients={"warehouse": warehouse})
    assert lineage.number(rebuilt) == {
        id(step): index for index, step in enumerate(lineage.walk(rebuilt))
    }


def test_structural_sharing_survives_the_round_trip(warehouse: Engine) -> None:
    """A view feeding two models is one node referenced twice, not two nodes."""
    original = _plan(warehouse)
    document = dump(original)

    assert len(document.steps) == len(lineage.walk(original))

    rebuilt = load(document, clients={"warehouse": warehouse})
    models = [step for step in lineage.walk(rebuilt) if isinstance(step, Model)]
    linkers = [model for model in models if model.right is not None]
    assert len(linkers) == 2
    # Both linkers hold the *same* view object, not two equal ones.
    assert linkers[0].left is linkers[1].left


def test_a_document_carries_no_client_or_credentials(warehouse: Engine) -> None:
    """Locations describe where data lives; connecting is the target's business."""
    document = dump(_plan(warehouse))
    serialised = document.model_dump_json()

    assert "sqlite" not in serialised  # the engine's URL never leaves
    assert '"name":"warehouse"' in serialised  # but the location it needs is named

    with pytest.raises(ValueError, match="has no client"):
        load(document, clients={})


# -- what the document says -----------------------------------------------------------


def test_edges_are_positions_pointing_backwards(warehouse: Engine) -> None:
    """Topological order is the document's ordering guarantee."""
    document = dump(_plan(warehouse))

    assert document.steps[0].kind == "source"
    assert document.steps[-1].kind == "resolver"
    for position, node in enumerate(document.steps):
        assert all(target < position for target in node.inputs)


def test_a_config_describes_settings_and_never_edges(warehouse: Engine) -> None:
    """The split that makes a config safe to hash and a document able to rebuild."""
    document = dump(_plan(warehouse))

    for node in document.steps:
        config = node.config.model_dump(mode="json")
        if node.kind == "clean":
            assert set(config) == {"cleaning", "group"}
        if node.kind == "model":
            assert set(config) == {"model_type", "model_class", "model_settings"}
        # A source's own name is settings, not an edge: it prefixes its columns.
        if node.kind == "source":
            assert config["name"] in {"crn", "dh"}


def test_a_setting_that_points_at_an_input_travels_as_a_position(
    warehouse: Engine,
) -> None:
    """Thresholds key by position, which has to survive JSON's string-only keys.

    A name would have been the alternative, and would have made the document depend on
    names being stable — the thing positions exist to avoid.
    """
    document = _transfer(dump(_plan(warehouse)))
    apex = document.steps[-1]

    assert apex.kind == "resolver"
    # `1` is the second input, and JSON has stringified the key.
    assert apex.config.resolver_settings == {"thresholds": {"1": 0.5}}

    rebuilt = load(document, clients={"warehouse": warehouse})
    assert rebuilt.resolver_settings.thresholds == {1: 0.5}


def test_a_document_rejects_an_edge_that_points_forwards() -> None:
    """Inputs before consumers is an invariant, not a convention."""
    with pytest.raises(ValueError, match="not an earlier step"):
        PlanDocument.model_validate(
            {
                "steps": [
                    {"kind": "clean", "config": {}, "inputs": [1]},
                    {"kind": "clean", "config": {}, "inputs": []},
                ]
            }
        )


def test_config_is_parsed_by_kind_not_guessed() -> None:
    """`ViewConfig` has no required fields, so a plain union would swallow anything."""
    document = PlanDocument.model_validate(
        {
            "steps": [
                {
                    "kind": "model",
                    "config": {
                        "model_type": "deduper",
                        "model_class": "NaiveDeduper",
                        "model_settings": {"unique_fields": ["company"]},
                    },
                    "inputs": [],
                }
            ]
        }
    )
    assert document.steps[0].config.model_class == "NaiveDeduper"


def test_load_rejects_a_step_wired_to_the_wrong_kind(warehouse: Engine) -> None:
    """A malformed document fails at load, not with a confusing error much later."""
    document = dump(_plan(warehouse))
    resolver = next(node for node in document.steps if node.kind == "resolver")
    broken = PlanDocument(
        steps=tuple(
            node.model_copy(update={"inputs": (0,)}) if node is resolver else node
            for node in document.steps
        )
    )

    with pytest.raises(ValueError, match="must read only"):
        load(broken, clients={"warehouse": warehouse})


def test_an_empty_document_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one step"):
        load(PlanDocument(steps=()), clients={})


def test_a_view_is_rebuilt_with_its_resolver(warehouse: Engine) -> None:
    """A view's inputs are sources plus at most one resolver, told apart by kind."""
    rebuilt = load(dump(_plan(warehouse)), clients={"warehouse": warehouse})
    views = [step for step in lineage.walk(rebuilt) if isinstance(step, View)]

    through = [view for view in views if view.resolver is not None]
    assert len(through) == 1
    assert isinstance(through[0].resolver, Resolver)
    assert [source.name for source in through[0].sources] == ["crn"]
