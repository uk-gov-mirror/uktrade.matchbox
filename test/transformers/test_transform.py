"""The `Transform` plan node, covering desugaring, construction, and the registry.

These assert on plan *shape*, so they build over a real `Source` from the `source`
fixture but never collect. The identities hold before any warehouse read.
"""

from collections.abc import Callable

import polars as pl
import pytest

from matchlab import Source
from matchlab.transformers import (
    Clean,
    Select,
    Transform,
    Transformer,
    add_transformer_class,
)


def test_select_desugars_to_transform(source: Callable[..., Source]) -> None:
    """`source.select(...)` is a transform carrying a `Select`."""
    crn = source("crn")
    step = crn.select("crn_company", "crn_town")

    assert isinstance(step, Transform)
    assert isinstance(step.transformer, Select)
    assert step.transformer.columns == ("crn_company", "crn_town")
    assert step.upstream == (crn,)


@pytest.mark.parametrize(
    ("verb", "explicit"),
    [
        pytest.param(
            lambda crn: crn.clean({"name": "lower(crn_company)"}),
            lambda crn: crn.transform(Clean(cleaning={"name": "lower(crn_company)"})),
            id="clean",
        ),
        pytest.param(
            lambda crn: crn.select("crn_company"),
            lambda crn: crn.transform(Select("crn_company")),
            id="select",
        ),
    ],
)
def test_verb_equals_transform_of_transformer(
    source: Callable[..., Source],
    verb: Callable[[Source], Transform],
    explicit: Callable[[Source], Transform],
) -> None:
    """The convenience verb is exactly `transform()` of the matching transformer."""
    crn = source("crn")
    assert verb(crn).spec == explicit(crn).spec


def test_transform_builds_from_name_and_dict(source: Callable[..., Source]) -> None:
    """A name plus a settings dict rebuilds the same transformer as an instance.

    This is the path `document.load` takes when reconstructing a plan.
    """
    crn = source("crn")
    from_dict = Transform(crn, "Clean", {"cleaning": {"name": "lower(crn_company)"}})
    from_instance = Transform(crn, Clean(cleaning={"name": "lower(crn_company)"}))

    assert from_dict.transformer == from_instance.transformer
    assert from_dict.spec == from_instance.spec


class _Double(Transformer):
    """A custom transformer that doubles one integer column, for the registry test."""

    column: str

    def apply(self, data: pl.DataFrame) -> pl.DataFrame:
        return data.with_columns((pl.col(self.column) * 2).alias(self.column))


def test_add_transformer_class_makes_it_nameable(
    source: Callable[..., Source],
) -> None:
    """A registered custom transformer can be named in a plan, as a built-in can."""
    add_transformer_class(_Double)
    crn = source("crn")

    step = Transform(crn, "_Double", {"column": "n"})

    assert isinstance(step.transformer, _Double)
    assert step.transformer.apply(pl.DataFrame({"n": [1, 2]}))["n"].to_list() == [2, 4]


def test_add_transformer_class_rejects_non_transformer() -> None:
    """The registry only accepts `Transformer` subclasses."""
    with pytest.raises(ValueError, match="not a subclass of Transformer"):
        add_transformer_class(str)
