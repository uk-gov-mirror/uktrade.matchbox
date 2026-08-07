"""Serialise a plan to a portable document, and rebuild it somewhere else.

A `PlanDocument` is a **derived view** of a plan, not a source format. Humans write
Python, and a document is what you hand to another environment so it can run the same
plan. It is dumped, transferred and loaded, never hand-authored. That is why nodes
refer to each other by position rather than by any human-facing name.

Nodes come out in `lineage.walk` order, so every input index is smaller than the index
of the step that consumes it. Positions also preserve structural sharing. A view
feeding two models is one node referenced twice. Nesting each step's inputs inside it
would have inlined the whole subtree twice over instead.

**Edges live here, settings live in the spec.** That split is the point of this
module. A spec is hashed into a fingerprint, so it must carry everything that changes
the step's output and nothing else. A serialised plan must also carry the edges, which
a fingerprint already covers by folding in its parents' fingerprints. Putting both in
one model would have forced a choice: a spec that lied about identity, where a rename
invalidates a subtree that produces identical bytes, or a document that could not be
rebuilt.

Edges are not the only thing that lands on this side of the split. A source's
`LocationRef` says which location class to build and under what name, so that `load`
can attach a client. Neither fact changes a byte the source produces, which is why a
`SourceSpec` records nothing about where its rows came from.

**What a document cannot carry:**

* *Clients.* A document names the locations its sources read and stops there. `load`
  takes the clients and attaches them by name, so credentials never enter a document.
* *Code.* `model_class` and `resolver_class` are registry names, so the target
  environment must have the same classes registered (`add_model_class`). A document is
  portable across environments, not across codebases.
* *Labels.* Publishing is something you do to a *result*, using `Resolver.publish`. It
  is not part of the plan, so the receiving environment collects and then publishes
  under whatever label it wants. The only names in a document are sources' names,
  which are part of their output.
* *Data, or any hash of it.* A source's fingerprint folds in a content hash of what it
  actually read, and that hash is derived on load from the target warehouse. Same rows
  give the same fingerprints, so the target store hits cache instead of recomputing.
  Different rows give different fingerprints, so it re-runs. Both are the intended
  behaviour.
"""

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from matchlab.core.kinds import StepKind
from matchlab.lineage import walk
from matchlab.locations import build_location
from matchlab.models import Model
from matchlab.resolvers import Resolver
from matchlab.sources import Source
from matchlab.specs import (
    ModelSpec,
    ResolverSpec,
    SourceSpec,
    ViewSpec,
)
from matchlab.steps import Step
from matchlab.views import View

StepSpec = SourceSpec | ViewSpec | ModelSpec | ResolverSpec

SpecT = TypeVar("SpecT", bound=BaseModel)
StepT = TypeVar("StepT", bound=Step)

_SPEC_TYPES: dict[StepKind, type[BaseModel]] = {
    StepKind.SOURCE: SourceSpec,
    StepKind.VIEW: ViewSpec,
    StepKind.MODEL: ModelSpec,
    StepKind.RESOLVER: ResolverSpec,
}


class LocationRef(BaseModel):
    """Everything needed to rebuild a source's location."""

    model_config = ConfigDict(frozen=True)

    location_class: str = Field(
        description="The registered name of the Location subclass to build."
    )
    name: str = Field(
        description="The key `load` looks up in `clients` to find this location's."
    )


class StepNode(BaseModel):
    """One step: what kind it is, what it specifies, and what it reads."""

    model_config = ConfigDict(frozen=True)

    kind: StepKind = Field(description="Which kind of step this is.")
    spec: StepSpec = Field(
        description="This step's own settings, exactly what its fingerprint hashes."
    )
    inputs: tuple[int, ...] = Field(
        default=(),
        description=(
            "Positions of this step's inputs, in order. Always smaller than this "
            "step's own position, and order matters: it is the order the fingerprint "
            "folds parents in, and the order a linker's left and right arrive in."
        ),
    )
    location: LocationRef | None = Field(
        default=None,
        description=(
            "For a source, how to rebuild the location it reads. Here rather than in "
            "the spec because it describes reconstruction rather than output. See "
            "`LocationRef`. Every other kind of step leaves it unset."
        ),
    )

    @model_validator(mode="after")
    def _check_location(self) -> "StepNode":
        """A source names a location. Nothing else has one to name."""
        if self.kind is StepKind.SOURCE and self.location is None:
            raise ValueError("A source step must name the location it reads.")
        if self.kind is not StepKind.SOURCE and self.location is not None:
            raise ValueError(f"A {self.kind} step must not name a location.")
        return self

    @model_validator(mode="before")
    @classmethod
    def _spec_from_kind(cls, data: Any) -> Any:  # noqa: ANN401 - raw pydantic input
        """Parse `spec` as the type `kind` names, rather than guessing.

        The four specs are structurally close enough that a plain union mis-assigns:
        `ViewSpec` has no required fields, so an empty object matches it and several
        others. `kind` is the discriminator the document already carries.
        """
        if not isinstance(data, dict):
            return data
        # `kind` is still whatever the input carried, a plain string, from JSON. It
        # finds its entry anyway, because a `StrEnum` member hashes and compares as
        # its value.
        kind, spec = data.get("kind"), data.get("spec")
        if isinstance(spec, dict) and kind in _SPEC_TYPES:
            return {**data, "spec": _SPEC_TYPES[kind].model_validate(spec)}
        return data


class PlanDocument(BaseModel):
    """A whole plan: its steps in topological order, and the edges between them."""

    model_config = ConfigDict(frozen=True)

    steps: tuple[StepNode, ...] = Field(
        description=(
            "Every step reachable from the plan's apex, inputs before consumers."
        )
    )

    @model_validator(mode="after")
    def _check_edges(self) -> "PlanDocument":
        """Reject a document whose edges do not point backwards at real steps."""
        for position, node in enumerate(self.steps):
            for target in node.inputs:
                if not 0 <= target < position:
                    raise ValueError(
                        f"Step {position} ({node.kind}) reads step {target}, which is "
                        "not an earlier step. Inputs must precede their consumers."
                    )
        return self


def dump(root: Step) -> PlanDocument:
    """Describe `root` and everything it reads as a portable document.

    Args:
        root: The apex of the plan. Only what is reachable upstream of it is included,
            exactly as `collect()` would run.

    Returns:
        The document. It holds no client, no credentials and no data.
    """
    ordered = walk(root)
    position = {id(step): index for index, step in enumerate(ordered)}
    return PlanDocument(
        steps=tuple(
            _node(step, tuple(position[id(parent)] for parent in step.upstream))
            for step in ordered
        )
    )


def _node(step: Step, inputs: tuple[int, ...]) -> StepNode:
    """Describe one step, checking its spec is the one its kind implies."""
    spec_type = _SPEC_TYPES.get(step.kind)
    if spec_type is None or not isinstance(step.spec, spec_type):
        raise ValueError(
            f"A {step.kind} step has no matching spec type. A new kind of step "
            "needs an entry in `_SPEC_TYPES`."
        )
    return StepNode(
        kind=step.kind,
        spec=cast(StepSpec, step.spec),
        inputs=inputs,
        location=(
            LocationRef(
                location_class=type(step.location).__name__, name=step.location.name
            )
            if isinstance(step, Source)
            else None
        ),
    )


def load(document: PlanDocument, clients: Mapping[str, Any]) -> Step:
    """Rebuild a plan from a document, returning its apex.

    Nothing is collected. This reconstructs the same lazy plan, so the returned step
    fingerprints identically to the one that was dumped, given the same data.

    Args:
        document: A dumped plan.
        clients: Location name to client, for every location the document's sources
            name. A `RelationalDBLocation` takes a SQLAlchemy engine or an ADBC
            connection.

    Returns:
        The plan's apex, the last step in the document.

    Raises:
        ValueError: If the document is empty, names a location with no client or an
            unregistered location class, or wires a step to an input of the wrong kind.
    """
    if not document.steps:
        raise ValueError("A plan document needs at least one step")

    built: list[Step] = []
    for position, node in enumerate(document.steps):
        built.append(_rebuild(node, position, [built[i] for i in node.inputs], clients))
    return built[-1]


def _rebuild(
    node: StepNode, position: int, inputs: list[Step], clients: Mapping[str, Any]
) -> Step:
    """Rebuild one step from its node and its already-rebuilt inputs."""
    match node.kind:
        case StepKind.SOURCE:
            spec = _expect(node, position, SourceSpec)
            reference = cast(LocationRef, node.location)
            if reference.name not in clients:
                raise ValueError(
                    f"Source '{spec.name}' reads location '{reference.name}', which "
                    f"has no client. Pass one in `clients`: "
                    f"{{'{reference.name}': engine}}."
                )
            return Source(
                location=build_location(
                    reference.location_class,
                    name=reference.name,
                    client=clients[reference.name],
                ),
                name=spec.name,
                extract_transform=spec.extract_transform,
                key_field=spec.key_field,
            )

        case StepKind.VIEW:
            spec = _expect(node, position, ViewSpec)
            sources = tuple(step for step in inputs if isinstance(step, Source))
            resolvers = [step for step in inputs if isinstance(step, Resolver)]
            if len(sources) + len(resolvers) != len(inputs) or len(resolvers) > 1:
                raise ValueError(
                    f"Step {position} (view) must read sources and at most one "
                    f"resolver, got {[step.kind for step in inputs]}."
                )
            return View(
                *sources,
                resolver=resolvers[0] if resolvers else None,
                cleaning=spec.cleaning,
                group=spec.group,
            )

        case StepKind.MODEL:
            spec = _expect(node, position, ModelSpec)
            views = _all_of(node, position, inputs, View)
            if not 1 <= len(views) <= 2:
                raise ValueError(
                    f"Step {position} (model) reads {len(views)} views; a deduper "
                    "reads one and a linker two."
                )
            return Model(
                left=views[0],
                right=views[1] if len(views) > 1 else None,
                model_class=spec.model_class,
                model_settings=spec.model_settings,
            )

        case StepKind.RESOLVER:
            spec = _expect(node, position, ResolverSpec)
            return Resolver(
                *_all_of(node, position, inputs, Model),
                resolver_class=spec.resolver_class,
                resolver_settings=spec.resolver_settings,
            )

        case _:  # every member of StepKind is handled above
            raise ValueError(
                f"No rebuild for a {node.kind} step. A new kind of step needs an arm "
                "here as well as an entry in `_SPEC_TYPES`."
            )


def _expect(node: StepNode, position: int, spec_type: type[SpecT]) -> SpecT:
    """Narrow a node's spec to the type its kind implies."""
    if not isinstance(node.spec, spec_type):
        raise ValueError(  # the validator parses spec by kind
            f"Step {position} ({node.kind}) has a "
            f"{type(node.spec).__name__}, expected {spec_type.__name__}."
        )
    return node.spec


def _all_of(
    node: StepNode, position: int, inputs: list[Step], step_type: type[StepT]
) -> tuple[StepT, ...]:
    """Assert every input is the kind this step can read, and return them in order."""
    checked = tuple(step for step in inputs if isinstance(step, step_type))
    if len(checked) != len(inputs):
        raise ValueError(
            f"Step {position} ({node.kind}) must read only "
            f"{step_type.__name__.lower()}s, got {[step.kind for step in inputs]}."
        )
    return checked
