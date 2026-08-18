"""Tests for the `Resource` wrapper itself.

What a resource does once it is inside a plan — the settings/resources split, the
fingerprint it must not reach, sharing one object between steps — belongs with the
plan, in `test/plan/` and `test_document.py`.
"""

import pytest
from sqlalchemy import Engine, create_engine

from matchlab import Resource
from matchlab.core.exceptions import ResourceError
from matchlab.resources import as_resources, names_of, values_of


@pytest.fixture
def engine() -> Engine:
    """A throwaway engine to stand in for anything a document cannot carry."""
    return create_engine("sqlite://")


def test_resource_empty_name(engine: Engine) -> None:
    """A name is the whole of what a document records, so it has to be one."""
    with pytest.raises(ResourceError, match="non-empty string"):
        Resource("", engine)
    with pytest.raises(ResourceError, match="non-empty string"):
        Resource(None, engine)


def test_resource_repr(engine: Engine) -> None:
    """A resource may hold a credential, so its repr shows the name and type only."""
    assert repr(Resource("warehouse", engine)) == "Resource('warehouse', <Engine>)"
    assert "anonymous" in repr(Resource.anonymous(engine))


def test_bare_values(engine: Engine) -> None:
    """A caller may pass a bare object; it is usable, and unnameable."""
    wrapped = as_resources({"client": engine, "other": Resource("named", engine)})

    assert wrapped["client"].is_anonymous
    assert wrapped["client"].value is engine
    assert not wrapped["other"].is_anonymous
    assert values_of(wrapped) == {"client": engine, "other": engine}
    # Only named resources can be recorded, which is how `dump` spots the other kind.
    assert names_of(wrapped) == {"other": "named"}
