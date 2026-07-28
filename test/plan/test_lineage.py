"""Unit tests for the lineage algorithms.

These exercise `matchlab.lineage` in isolation, with trivial fake steps, so the
graph semantics are pinned independently of sources, adapters, or warehouses.
"""

from typing import ClassVar

from pydantic import BaseModel

from matchlab import lineage
from matchlab.adapters import Adapter, Fingerprint
from matchlab.steps import Step


class _FakeConfig(BaseModel):
    """Minimal config so FakeStep satisfies the Step contract."""

    name: str


class FakeStep(Step):
    """A plan node that computes nothing.

    Carries a `label` purely so these tests can tell nodes apart. Real steps have no
    such thing — they are identified by position.
    """

    kind: ClassVar[str] = "fake"

    def __init__(self, label: str = "fake", upstream: tuple[Step, ...] = ()) -> None:
        self.label = label
        super().__init__(upstream=upstream)

    def __str__(self) -> str:
        return self.label

    @property
    def config(self) -> BaseModel:
        return _FakeConfig(name=self.label)

    def _execute(self, adapter: Adapter, fp: Fingerprint) -> None:  # pragma: no cover
        raise AssertionError("FakeStep never executes")


def test_walk_returns_inputs_before_consumers() -> None:
    a = FakeStep("a")
    b = FakeStep("b", upstream=(a,))
    c = FakeStep("c", upstream=(b,))

    order = [step.label for step in lineage.walk(c)]
    assert order == ["a", "b", "c"]


def test_walk_visits_a_shared_input_once() -> None:
    """A diamond: shared nodes are structurally shared, not duplicated."""
    shared = FakeStep("shared")
    left = FakeStep("left", upstream=(shared,))
    right = FakeStep("right", upstream=(shared,))
    root = FakeStep("root", upstream=(left, right))

    order = [step.label for step in lineage.walk(root)]
    assert order.count("shared") == 1
    # Still topologically valid: the shared input precedes both consumers.
    assert order.index("shared") < order.index("left")
    assert order.index("shared") < order.index("right")
    assert order[-1] == "root"


def test_walk_from_a_leaf_is_just_the_leaf() -> None:
    leaf = FakeStep("leaf")
    FakeStep("downstream", upstream=(leaf,))  # never reachable from the leaf
    assert [step.label for step in lineage.walk(leaf)] == ["leaf"]


def test_draw_expands_a_shared_node_once_and_refers_back_after() -> None:
    """Otherwise the drawing contradicts the sharing it exists to show.

    Redrawing a shared subtree per branch is exponential in the depth of the sharing —
    the `examples/companies` plan is 24 steps and 187 expanded lines. A position makes
    the back-reference readable: `↑ [0]` names a node you have already seen.
    """
    shared = FakeStep("shared")
    left = FakeStep("left", upstream=(shared,))
    right = FakeStep("right", upstream=(shared,))
    root = FakeStep("root", upstream=(left, right))

    drawing = lineage.draw(root)

    assert drawing.count("shared") == 2  # once expanded, once referred to
    assert drawing.count("shared ↑") == 1
    # Every node is still named at its own position, which is what the log quotes.
    for position in lineage.number(root).values():
        assert f"[{position}] " in drawing


def test_draw_nests_inputs_under_consumers() -> None:
    source = FakeStep("source")
    apex = FakeStep("apex", upstream=(source,))

    drawing = lineage.draw(apex)
    assert "apex" in drawing
    assert "source" in drawing
    # The input is indented beneath its consumer.
    assert drawing.index("apex") < drawing.index("source")
    assert "└── " in drawing


def test_draw_numbers_nodes_by_walk_position_not_by_line() -> None:
    """The drawing is the key to the log: `step 7` in a run is `[7]` here.

    A tree prints consumers above their inputs while a walk lists inputs first, so the
    numbers deliberately do not ascend down the page.
    """
    source = FakeStep("source")
    apex = FakeStep("apex", upstream=(source,))

    drawing = lineage.draw(apex)
    assert "[1] apex" in drawing  # apex is walked last
    assert "[0] source" in drawing
    assert drawing.index("[1]") < drawing.index("[0]")


def test_numbering_writes_nothing_back_onto_the_steps() -> None:
    """A position belongs to the walk, not to the step.

    The same node numbers differently depending on the apex, so storing one on the step
    would make it a lie as soon as anything walked from somewhere else. Callers that
    need a position are handed it.
    """
    source = FakeStep()
    apex = FakeStep(upstream=(source,))

    assert lineage.number(apex) == {id(source): 0, id(apex): 1}
    assert lineage.number(source) == {id(source): 0}
    assert not hasattr(source, "_position")
    assert not hasattr(source, "name")  # steps have no names at all


def test_steps_have_no_downstream_reference() -> None:
    """The structural guarantee: a node exposes its inputs and nothing else."""
    source = FakeStep("source")
    FakeStep("apex", upstream=(source,))

    assert source.upstream == ()
    assert not hasattr(source, "downstream")
    assert not hasattr(source, "dag")
