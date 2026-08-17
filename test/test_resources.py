"""Tests for Resource."""

import polars as pl
import pytest
from pydantic import ConfigDict
from sqlalchemy import Engine, create_engine

from matchlab import (
    FromResources,
    Resource,
    Source,
    dump,
    load,
    read_database,
    read_dataframe,
)
from matchlab.adapters import DuckDBAdapter
from matchlab.core.exceptions import ResourceError
from matchlab.core.kinds import StepKind
from matchlab.models.dedupers.base import Deduper
from matchlab.resolvers import Resolver
from matchlab.resolvers.components import Components
from matchlab.resources import as_resources, names_of, values_of
from matchlab.transformers import Select


@pytest.fixture
def engine() -> Engine:
    """A throwaway engine to stand in for anything a document cannot carry."""
    return create_engine("sqlite://")


# -- the wrapper ----------------------------------------------------------------------


def test_resource_rejects_empty_name(engine: Engine) -> None:
    """A name is the whole of what a document records, so it has to be one."""
    with pytest.raises(ResourceError, match="non-empty string"):
        Resource("", engine)
    with pytest.raises(ResourceError, match="non-empty string"):
        Resource(None, engine)


def test_resource_repr_hides_the_value(engine: Engine) -> None:
    """A resource may hold a credential, so its repr shows the name and type only."""
    assert repr(Resource("warehouse", engine)) == "Resource('warehouse', <Engine>)"
    assert "anonymous" in repr(Resource.anonymous(engine))


def test_bare_values_are_wrapped(engine: Engine) -> None:
    """A caller may pass a bare object; it is usable, and unnameable."""
    wrapped = as_resources({"client": engine, "other": Resource("named", engine)})

    assert wrapped["client"].is_anonymous
    assert wrapped["client"].value is engine
    assert not wrapped["other"].is_anonymous
    assert values_of(wrapped) == {"client": engine, "other": engine}
    # Only named resources can be recorded, which is how `dump` spots the other kind.
    assert names_of(wrapped) == {"other": "named"}


# -- in a plan ------------------------------------------------------------------------


def test_settings_vs_resources(engine: Engine) -> None:
    """The query is specified, the client is named."""
    source = read_database(
        "ch", sql="select id, c from ch", client=Resource("wh", engine)
    )
    node = dump(source).steps[0]

    assert node.spec.location_settings == {"sql": "select id, c from ch"}
    assert node.resources == {"client": "wh"}
    assert "engine" not in node.spec.model_dump_json()


def test_renaming_resource(warehouse: Engine, adapter: DuckDBAdapter) -> None:
    """Renaming a resource keeps the fingerprint stable."""

    def apex(resource_name: str) -> Resolver:
        crn = read_database(
            "crn",
            sql="select pk, company from crn",
            client=Resource(resource_name, warehouse),
            key_field="pk",
        )
        return crn.dedupe(
            model_class="NaiveDeduper",
            model_settings={"unique_fields": ["crn_company"]},
        ).resolve()

    original = apex("warehouse").collect(adapter, interactive=False)
    renamed = apex("something_else_entirely").collect(adapter, interactive=False)

    assert renamed.fingerprints() == original.fingerprints()


def test_one_name_two_objects(warehouse: Engine) -> None:
    """Two different objects cannot be tied to the same resource name.

    Two engines under one name would load as one, silently changing the plan.
    """
    other = create_engine("sqlite://")
    left = read_database(
        "crn", sql="select pk, company from crn", client=Resource("wh", warehouse)
    )
    right = read_database(
        "dh", sql="select pk, company from dh", client=Resource("wh", other)
    )
    plan = left.link(
        right,
        model_class="DeterministicLinker",
        model_settings={"comparisons": ["l.crn_company = r.dh_company"]},
    ).resolve()

    with pytest.raises(ResourceError, match="covers two different objects"):
        dump(plan)


def test_resource_sharing(warehouse: Engine) -> None:
    """Two locations can share one warehouse via resources."""
    client = Resource("wh", warehouse)
    left = read_database("crn", sql="select pk, company from crn", client=client)
    right = read_database("dh", sql="select pk, company from dh", client=client)
    plan = left.link(
        right,
        model_class="DeterministicLinker",
        model_settings={"comparisons": ["l.crn_company = r.dh_company"]},
    ).resolve()

    assert dump(plan).required_resources() == {"wh"}


def test_bare_value(warehouse: Engine) -> None:
    """A bare value can be used instead of a resource, until dump is attemped."""
    source = read_database("crn", sql="select pk, company from crn", client=warehouse)

    assert source.sample().height > 0  # usable here

    with pytest.raises(ResourceError, match="unnamed resource for 'client'"):
        dump(source)


def test_required_resources_list(warehouse: Engine) -> None:
    """A caller can see what to supply before trying to load."""
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

    document = dump(plan)
    assert document.required_resources() == {"wh", "dh_frame"}

    with pytest.raises(ResourceError, match="needs resource 'dh_frame'"):
        load(document, resources={"wh": warehouse})


# No built-in methodology takes a resource, so a test for one has to define them.


class NeedyDeduper(Deduper):
    """A deduper holding a connection it never reads through."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    unique_fields: list[str]
    session: FromResources[Engine]

    def prepare(self, data: pl.DataFrame) -> None:
        """Never called: this plan is inspected, not collected."""

    def dedupe(self, data: pl.DataFrame) -> pl.DataFrame:
        """Never called: this plan is inspected, not collected."""
        raise NotImplementedError


class NeedySelect(Select):
    """A transformer holding one."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    session: FromResources[Engine]


class NeedyComponents(Components):
    """A resolver methodology holding one."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session: FromResources[Engine]


#: Where each kind of spec keeps its serialised settings.
SETTINGS_FIELD = {
    StepKind.SOURCE: "location_settings",
    StepKind.TRANSFORM: "transformer_settings",
    StepKind.MODEL: "model_settings",
    StepKind.RESOLVER: "resolver_settings",
}


def test_resource_not_in_spec(warehouse: Engine) -> None:
    """A step records its resources beside its spec, never inside it."""
    session = Resource("session", warehouse)
    plan = (
        read_database(
            "crn",
            sql="select pk, company from crn",
            client=Resource("wh", warehouse),
            key_field="pk",
        )
        .transform(NeedySelect, {"columns": ("crn_company",)}, {"session": session})
        .dedupe(NeedyDeduper, {"unique_fields": ["crn_company"]}, {"session": session})
        .resolve(
            resolver_class=NeedyComponents, resolver_resources={"session": session}
        )
    )

    for node in dump(plan).steps:
        settings = getattr(node.spec, SETTINGS_FIELD[node.kind])

        assert node.resources, f"{node.kind} bound no resource"
        assert not set(node.resources) & set(settings), (
            f"{node.kind} leaked a resource field into its settings"
        )


# -- which argument a field belongs in ------------------------------------------------


def test_resource_marked_field(warehouse: Engine) -> None:
    """A marked field must be a resource."""
    with pytest.raises(ResourceError, match="'client' on RelationalDB is marked"):
        Source(
            location_class="RelationalDB",
            name="crn",
            location_settings={"sql": "select pk from crn", "client": warehouse},
        )


def test_resource_unmarked_field(warehouse: Engine) -> None:
    """An unmarked field must be a setting."""
    source = read_database("crn", sql="select pk, company from crn", client=warehouse)

    with pytest.raises(ResourceError, match="'columns' on Select is a setting"):
        source.transform(Select, transformer_resources={"columns": ("crn_company",)})


def test_unknown_field(warehouse: Engine) -> None:
    """A field the methodology doesn't have says so."""
    source = read_database("crn", sql="select pk, company from crn", client=warehouse)

    with pytest.raises(ResourceError, match="Select has no field 'nonexistent'"):
        source.transform(Select, transformer_resources={"nonexistent": 1})
