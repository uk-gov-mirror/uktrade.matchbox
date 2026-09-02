"""Unit tests for the lineage algorithms.

These exercise `matchlab.lineage` in isolation, with trivial fake steps, so the
graph semantics are pinned independently of sources, stores, or warehouses.
"""

from typing import ClassVar

from pydantic import BaseModel

from matchlab import lineage
from matchlab.core.kinds import StepKind
from matchlab.steps import Step
from matchlab.stores import Fingerprint, Store


class _FakeSpec(BaseModel):
    """Minimal spec so FakeStep satisfies the Step contract."""

    name: str


class FakeStep(Step):
    """A plan node that computes nothing.

    Carries a `label` purely so these tests can tell nodes apart, and overrides
    `__str__` to draw it. Real steps have no such thing — they are identified by
    position, and drawn by kind. The kind itself is arbitrary here: nothing in
    `lineage` reads it, and `Step.kind` admits only real ones.
    """

    kind: ClassVar[StepKind] = StepKind.TRANSFORM

    def __init__(self, label: str = "fake", parents: tuple[Step, ...] = ()) -> None:
        """Create a fake step with the given label and parent steps."""
        self.label = label
        self._parents = parents
        super().__init__()

    @property
    def parents(self) -> tuple[Step, ...]:
        """`Step` leaves each kind to name its own inputs; this one just holds them."""
        return self._parents

    def __str__(self) -> str:
        """Return the label, so drawings identify nodes by it."""
        return self.label

    @property
    def spec(self) -> BaseModel:
        """Return a minimal spec carrying only the label."""
        return _FakeSpec(name=self.label)

    def _execute(self, store: Store, fp: Fingerprint) -> None:
        raise AssertionError("FakeStep never executes")

    def _methodology_class(self) -> type | None:
        """FakeStep runs no methodology; it computes nothing."""
        return None


def test_walk_inputs_before_consumers() -> None:
    """Walk yields every input before the step that consumes it."""
    a = FakeStep("a")
    b = FakeStep("b", parents=(a,))
    c = FakeStep("c", parents=(b,))

    order = [step.label for step in lineage.walk(c)]
    assert order == ["a", "b", "c"]


def test_walk_shared_once() -> None:
    """A diamond: shared nodes are structurally shared, not duplicated."""
    shared = FakeStep("shared")
    left = FakeStep("left", parents=(shared,))
    right = FakeStep("right", parents=(shared,))
    root = FakeStep("root", parents=(left, right))

    order = [step.label for step in lineage.walk(root)]
    assert order.count("shared") == 1
    # Still topologically valid: the shared input precedes both consumers.
    assert order.index("shared") < order.index("left")
    assert order.index("shared") < order.index("right")
    assert order[-1] == "root"


def test_walk_from_leaf() -> None:
    """Walk from a leaf is just the leaf."""
    leaf = FakeStep("leaf")
    FakeStep("downstream", parents=(leaf,))  # never reachable from the leaf
    assert [step.label for step in lineage.walk(leaf)] == ["leaf"]


def test_draw_shared_refers_back() -> None:
    """Otherwise the drawing contradicts the sharing it exists to show.

    Redrawing a shared subtree per branch is exponential in the depth of the sharing —
    the `examples/companies` plan is 24 steps and 187 expanded lines. A position makes
    the back-reference readable: `↑ [0]` names a node you have already seen.
    """
    shared = FakeStep("shared")
    left = FakeStep("left", parents=(shared,))
    right = FakeStep("right", parents=(shared,))
    root = FakeStep("root", parents=(left, right))

    drawing = lineage.draw(root)

    assert drawing.count("shared") == 2  # once expanded, once referred to
    assert drawing.count("shared ↑") == 1
    # Every node is still named at its own position, which is what the log quotes.
    for position in lineage.number(root).values():
        assert f"[{position}] " in drawing


def test_draw_nests_inputs() -> None:
    """Draw nests each step's inputs beneath it."""
    source = FakeStep("source")
    apex = FakeStep("apex", parents=(source,))

    drawing = lineage.draw(apex)
    assert "apex" in drawing
    assert "source" in drawing
    # The input is indented beneath its consumer.
    assert drawing.index("apex") < drawing.index("source")
    assert "└── " in drawing


def test_draw_numbers_by_position() -> None:
    """The drawing is the key to the log: `step 7` in a run is `[7]` here.

    A tree prints consumers above their inputs while a walk lists inputs first, so the
    numbers deliberately do not ascend down the page.
    """
    source = FakeStep("source")
    apex = FakeStep("apex", parents=(source,))

    drawing = lineage.draw(apex)
    assert "[1] apex" in drawing  # apex is walked last
    assert "[0] source" in drawing
    assert drawing.index("[1]") < drawing.index("[0]")


def test_number_is_read_only() -> None:
    """A position belongs to the walk, not to the step.

    The same node numbers differently depending on the apex, so storing one on the step
    would make it a lie as soon as anything walked from somewhere else. Callers that
    need a position are handed it.
    """
    source = FakeStep()
    apex = FakeStep(parents=(source,))

    assert lineage.number(apex) == {id(source): 0, id(apex): 1}
    assert lineage.number(source) == {id(source): 0}
    assert not hasattr(source, "_position")
    assert not hasattr(source, "name")  # steps have no names at all


def test_step_no_downstream_ref() -> None:
    """The structural guarantee: a node exposes its inputs and nothing else."""
    source = FakeStep("source")
    FakeStep("apex", parents=(source,))

    assert source.parents == ()
    assert not hasattr(source, "downstream")
    assert not hasattr(source, "dag")
