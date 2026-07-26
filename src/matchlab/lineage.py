"""Lineage algorithms over a plan tree.

Steps hold references to their *inputs* only (`step.upstream`) — there is no
registry and no downstream pointer, exactly as in Polars' logical plan. "The DAG"
is therefore whatever is reachable upstream from the node you are holding, and
every graph operation is a pure function of a root node.

Nodes are deduplicated by **object identity**, not by name or config: a node feeding
two branches is one object and is visited once (structural sharing), while two
structurally identical but distinct nodes are two plan entries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matchlab.steps import Step


def walk(root: Step) -> list[Step]:
    """Return `root` and every transitive input, in topological order.

    Upstream steps always precede the steps that consume them, so executing the
    returned list in order satisfies every dependency. Implemented as an iterative
    depth-first post-order over the upstream references.
    """
    ordered: list[Step] = []
    seen: set[int] = set()
    # (step, expanded) — `expanded` marks the second visit, when the node's inputs
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
        for parent in step.upstream:
            if id(parent) not in seen:
                stack.append((parent, False))

    return ordered


def number(root: Step) -> dict[int, int]:
    """Map each step in `root`'s plan to the position it is known by.

    A step is referred to by position — in logs, in `draw()`, and in a document.
    The position belongs to the walk rather than to the step: the
    same node numbers differently in `walk(deduped)` and `walk(companies)`, so nothing
    is written back onto the steps. Callers that need one hold this mapping, or are
    handed the number directly, as `collect` hands it to `_ensure`.

    Returns:
        Step identity to position, in walk order.
    """
    return {id(step): position for position, step in enumerate(walk(root))}


def draw(root: Step) -> str:
    """Render `root`'s sub-plan as an indented tree, inputs nested beneath consumers.

    Each node carries its **position** — the index `collect` runs it at, and the index
    it occupies in this plan's document. That is the cross-reference: a log line saying
    `step 7` is the node drawn as `[7]`. Positions come from `walk`, not from the order
    these lines happen to be printed in, since a tree nests consumers above their
    inputs while a walk lists inputs first.
    """
    position = number(root)
    lines: list[str] = []

    def render(step: Step, prefix: str, connector: str) -> None:
        marker = "●" if step.is_collected else "○"
        lines.append(f"{prefix}{connector}{marker} [{position[id(step)]}] {step}")
        child_prefix = prefix + ("    " if connector in ("└── ", "") else "│   ")
        parents = step.upstream
        for index, parent in enumerate(parents):
            last = index == len(parents) - 1
            render(parent, child_prefix, "└── " if last else "├── ")

    render(root, "", "")
    return "\n".join(lines)
