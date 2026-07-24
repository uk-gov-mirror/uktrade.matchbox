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

from matchlab.core.exceptions import StepNotFound

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


def find(root: Step, name: str) -> Step:
    """Return the step called `name` from `root`'s lineage.

    Raises:
        StepNotFound: If no step in the lineage has that name.
    """
    for step in walk(root):
        if step.name == name:
            return step
    raise StepNotFound(f"'{name}' is not in the lineage of '{root.name}'")


def validate(root: Step) -> None:
    """Check a plan is well-formed before it is executed.

    Cycles cannot be constructed (a step's inputs must exist before it does), so the
    only real hazard is two distinct steps sharing a name, which would make
    `find` — and any serialised form of the plan — ambiguous.

    Raises:
        ValueError: If two distinct steps in the lineage share a name.
    """
    by_name: dict[str, Step] = {}
    for step in walk(root):
        existing = by_name.get(step.name)
        if existing is not None and existing is not step:
            raise ValueError(
                f"Duplicate step name '{step.name}' in the lineage of '{root.name}'. "
                "Names must be unique within a plan."
            )
        by_name[step.name] = step


def draw(root: Step) -> str:
    """Render `root`'s sub-plan as an indented tree, inputs nested beneath consumers."""
    lines: list[str] = []

    def render(step: Step, prefix: str, connector: str) -> None:
        marker = "●" if step.is_collected else "○"
        lines.append(f"{prefix}{connector}{marker} {step.name} [{step.kind}]")
        child_prefix = prefix + ("    " if connector in ("└── ", "") else "│   ")
        parents = step.upstream
        for index, parent in enumerate(parents):
            last = index == len(parents) - 1
            render(parent, child_prefix, "└── " if last else "├── ")

    render(root, "", "")
    return "\n".join(lines)
