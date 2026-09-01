"""Lineage algorithms over a plan tree.

Steps hold references to their *inputs* only (`step.parents`). There is no registry
and no downstream pointer. "The DAG" is therefore
whatever is reachable upstream from the node you are holding, and every graph
operation is a pure function of a root node.

Nodes are deduplicated by **object identity**, not by name or config. A node feeding
two branches is one object and is visited once (structural sharing), while two
structurally identical but distinct nodes are two plan entries.

`draw` also owns the vocabulary for *how a step is doing*, meaning `StepStatus` and
`StepState`, because the tree is where that vocabulary is read. `matchlab.progress`
drives it during a collection. Nothing here executes anything.
"""

from collections.abc import Mapping
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    # `matchlab.steps` imports this module
    from matchlab.steps import Step


class StepStatus(StrEnum):
    """Where a step has got to in a collection.

    `CACHED`, `DONE` and `REFRESHED` all mean the artifact is available. They differ
    in why.

    `DONE` and `REFRESHED` both paid for the artifact. A `DONE` step ran because
    something it depends on moved. A `REFRESHED` step ran because it cannot be cached
    at all, and it will run again next time.
    """

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    CACHED = "cached"
    REFRESHED = "refreshed"
    FAILED = "failed"

    @property
    def marker(self) -> str:
        """A single-width glyph for this status.

        Deliberately geometric rather than emoji: emoji are double-width in some
        terminals and single in others, which knocks a tree's guide lines out of
        alignment line by line.
        """
        return _MARKERS[self]

    @property
    def style(self) -> str:
        """A Rich style for this status, used when drawing with markup."""
        return _STYLES[self]


_MARKERS: dict[StepStatus, str] = {
    StepStatus.PENDING: "○",
    StepStatus.RUNNING: "◐",
    StepStatus.DONE: "●",
    StepStatus.CACHED: "◍",
    StepStatus.REFRESHED: "↻",
    StepStatus.FAILED: "✗",
}

_STYLES: dict[StepStatus, str] = {
    StepStatus.PENDING: "dim",
    StepStatus.RUNNING: "bold yellow",
    StepStatus.DONE: "bold green",
    StepStatus.CACHED: "cyan",
    StepStatus.REFRESHED: "bold magenta",
    StepStatus.FAILED: "bold red",
}


# Statuses where an elapsed time is worth showing, the ones that spent any.
_TIMED = (
    StepStatus.RUNNING,
    StepStatus.DONE,
    StepStatus.REFRESHED,
    StepStatus.FAILED,
)


class StepState(BaseModel):
    """A step's status, and how long it has been in it."""

    model_config = ConfigDict(frozen=True)

    status: StepStatus
    elapsed: float | None = None

    def annotation(self) -> str:
        """The trailing detail shown after a step in a drawing, possibly empty.

        A time is shown only where it was spent. A cached step reads `cached` alone,
        because `0.0s` next to it says nothing.
        """
        parts: list[str] = []
        if self.status not in (StepStatus.PENDING, StepStatus.DONE):
            parts.append(self.status.value)
        if self.elapsed is not None and self.status in _TIMED:
            parts.append(f"{self.elapsed:.1f}s")
        return " ".join(parts)


def walk(root: "Step") -> list["Step"]:
    """Return `root` and every transitive input, in topological order.

    Upstream steps always precede the steps that consume them, so executing the
    returned list in order satisfies every dependency. Implemented as an iterative
    depth-first post-order over the parent references.
    """
    ordered: list[Step] = []
    seen: set[int] = set()
    # (step, expanded). `expanded` marks the second visit, when the node's inputs
    # have already been pushed and it is safe to emit.
    stack: list[tuple[Step, bool]] = [(root, False)]

    while stack:
        step, expanded = stack.pop()
        if expanded:
            ordered.append(step)
            continue
        if id(step) in seen:
            continue
        seen.add(id(step))
        stack.append((step, True))
        for parent in step.parents:
            if id(parent) not in seen:
                stack.append((parent, False))

    return ordered


def number(root: "Step") -> dict[int, int]:
    """Map each step in `root`'s plan to the position it is known by.

    A step is referred to by position, in logs, in `draw()`, and in a document.
    The position belongs to the walk rather than to the step: the
    same node numbers differently in `walk(deduped)` and `walk(companies)`, so nothing
    is written back onto the steps. Callers that need one hold this mapping, as
    `draw` does and as `matchlab.progress.Progress` does for the walk `collect` gave it.

    Returns:
        Step identity to position, in walk order.
    """
    return {id(step): position for position, step in enumerate(walk(root))}


def draw(
    root: "Step",
    state: Mapping[int, StepState] | None = None,
    *,
    markup: bool = False,
) -> str:
    """`root`'s sub-plan as an indented tree. See `render`, which this joins up."""
    return "\n".join(render(root, state, markup=markup)[0])


def render(
    root: "Step",
    state: Mapping[int, StepState] | None = None,
    *,
    markup: bool = False,
) -> tuple[list[str], dict[int, int]]:
    """Render `root`'s sub-plan as an indented tree, inputs nested beneath consumers.

    Each node carries its **position**, the index `collect` runs it at, and the index
    it occupies in this plan's document. The index is the cross-reference, where
    `step 7` is the node drawn as `[7]`. Positions come from `walk`, not from the order
    these lines happen to be printed in, since a tree nests consumers above their
    inputs while a walk lists inputs first.

    Args:
        root: The step to draw, along with everything upstream of it.
        state: Per-step status, keyed by `id(step)` as `number` is. Steps missing from
            the mapping are drawn as `PENDING`. When omitted, each step is drawn from
            its own `is_collected` flag instead, a static view of a plan you hold.
        markup: Emit Rich markup, styling each marker and annotation by status. Off by
            default, so the return value stays printable anywhere.

    Returns:
        The lines, one per node plus one per repeat reference, and the row each step
        is first drawn on, keyed by `id(step)` as `number` is. The rows are what let a
        caller show part of a tree and still know where it is: `matchlab.progress`
        windows a plan too tall for the terminal around the row that is running.

    A node feeding several branches is expanded where it is first met and marked `↑`
    everywhere after, rather than having its whole subtree redrawn each time. Positions
    are what make that readable. `↑ [12]` names the node you already have, and without
    it the drawing contradicts the structural sharing it is meant to show. The
    24-step plan in `examples/companies` draws 187 lines expanded, and 39 like this.
    """
    position = number(root)
    lines: list[str] = []
    expanded: set[int] = set()
    # Where each step is drawn. A repeat reference is a pointer to the expansion, not
    # another copy of it, so the first row is the one worth looking at.
    rows: dict[int, int] = {}

    def draw_step(step: "Step", prefix: str, connector: str) -> None:
        if state is None:
            marker = "●" if step.is_collected else "○"
            style, annotation = "", ""
        else:
            step_state = state.get(id(step), StepState(status=StepStatus.PENDING))
            marker = step_state.status.marker
            style = step_state.status.style
            annotation = step_state.annotation()

        repeat = id(step) in expanded
        expanded.add(id(step))

        label = f"[{position[id(step)]}] {step}"
        if annotation:
            label = f"{label} {annotation}"
        if repeat:
            label = f"{label} ↑"

        # Escape the whole label, not just the step. `[7]` would itself parse as a
        # style tag.
        row = (
            f"[{style}]{marker}[/] {_escape(label)}" if markup else f"{marker} {label}"
        )

        rows.setdefault(id(step), len(lines))
        lines.append(f"{prefix}{connector}{row}")
        if repeat:  # already drawn in full above, the reference is the whole line
            return

        child_prefix = prefix + ("    " if connector in ("└── ", "") else "│   ")
        parents = step.parents
        for index, parent in enumerate(parents):
            last = index == len(parents) - 1
            draw_step(parent, child_prefix, "└── " if last else "├── ")

    draw_step(root, "", "")
    return lines, rows


def _escape(text: str) -> str:
    """Escape Rich markup, so a `[7]` isn't read as a style tag."""
    return text.replace("[", r"\[")
