"""Unit tests for the lineage algorithms.

These exercise `matchlab.lineage` in isolation, with trivial fake steps, so the
graph semantics are pinned independently of sources, adapters, or warehouses.
"""

from typing import ClassVar

import pytest
from pydantic import BaseModel

from matchlab import lineage
from matchlab.adapters import Adapter, Fingerprint
from matchlab.steps import Step


class _FakeConfig(BaseModel):
    """Minimal config so FakeStep satisfies the Step contract."""

    name: str


class FakeStep(Step):
    """A plan node that computes nothing."""

    kind: ClassVar[str] = "fake"
    stores: ClassVar[bool] = False

    @property
    def config(self) -> BaseModel:
        return _FakeConfig(name=self.name)

    def _execute(self, adapter: Adapter, fp: Fingerprint) -> None:  # pragma: no cover
        raise AssertionError("FakeStep never executes")


def test_walk_returns_inputs_before_consumers() -> None:
    a = FakeStep("a")
    b = FakeStep("b", upstream=(a,))
    c = FakeStep("c", upstream=(b,))

    order = [step.name for step in lineage.walk(c)]
    assert order == ["a", "b", "c"]


def test_walk_visits_a_shared_input_once() -> None:
    """A diamond: shared nodes are structurally shared, not duplicated."""
    shared = FakeStep("shared")
    left = FakeStep("left", upstream=(shared,))
    right = FakeStep("right", upstream=(shared,))
    root = FakeStep("root", upstream=(left, right))

    order = [step.name for step in lineage.walk(root)]
    assert order.count("shared") == 1
    # Still topologically valid: the shared input precedes both consumers.
    assert order.index("shared") < order.index("left")
    assert order.index("shared") < order.index("right")
    assert order[-1] == "root"


def test_walk_from_a_leaf_is_just_the_leaf() -> None:
    leaf = FakeStep("leaf")
    FakeStep("downstream", upstream=(leaf,))  # never reachable from the leaf
    assert [step.name for step in lineage.walk(leaf)] == ["leaf"]


def test_validate_rejects_duplicate_names() -> None:
    a = FakeStep("dup")
    b = FakeStep("dup")  # distinct object, same name
    root = FakeStep("root", upstream=(a, b))

    with pytest.raises(ValueError, match="Duplicate step name"):
        lineage.validate(root)


def test_validate_accepts_the_same_node_reached_twice() -> None:
    """Sharing one node across branches is legal; only distinct clashes are not."""
    shared = FakeStep("shared")
    left = FakeStep("left", upstream=(shared,))
    right = FakeStep("right", upstream=(shared,))
    lineage.validate(FakeStep("root", upstream=(left, right)))


def test_draw_nests_inputs_under_consumers() -> None:
    source = FakeStep("source")
    apex = FakeStep("apex", upstream=(source,))

    drawing = lineage.draw(apex)
    assert "apex" in drawing
    assert "source" in drawing
    # The input is indented beneath its consumer.
    assert drawing.index("apex") < drawing.index("source")
    assert "└── " in drawing


def test_steps_have_no_downstream_reference() -> None:
    """The structural guarantee: a node exposes its inputs and nothing else."""
    source = FakeStep("source")
    FakeStep("apex", upstream=(source,))

    assert source.upstream == ()
    assert not hasattr(source, "downstream")
    assert not hasattr(source, "dag")
