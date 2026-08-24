"""Serialise a plan to a portable document, and rebuild it somewhere else.

A `PlanDocument` is a **derived view** of a plan, not a source format. Humans write
Python, and a document is what you hand to another environment so it can run the same
plan. It is dumped, transferred and loaded, never hand-authored. That is why nodes
refer to each other by position rather than by any human-facing name.

Nodes come out in `lineage.walk` order, so every input index is smaller than the index
of the step that consumes it. Positions also preserve structural sharing. A record step
feeding two models is one node referenced twice. Nesting each step's inputs inside it
would have inlined the whole subtree twice over instead.

**A node is three things, and they are separate for a reason.** Its `spec` is what the
fingerprint hashes, so it carries everything that changes the step's output and nothing
else. Its `inputs` are the edges, which a fingerprint already covers by folding in its
parents' fingerprints — recording them in the spec too would mean a rename invalidating
a subtree that produces identical bytes. Its `resources` name what the step reads
*through*, which changes no output and so belongs in neither.

Putting any two of them in one field would force a choice between a spec that lies about
identity and a document that cannot be rebuilt.

**What a document cannot carry:**

* *Resources.* Engines, connections, credentials, frames. A document records only the
  **name** each one was given (`matchlab.resources.Resource`), in `StepNode.resources`,
  keyed by the settings field it fills. `load` takes the real objects and fills them in,
  so nothing secret ever enters a document. Because they live in their own field rather
  than among the settings, no resource can reach a fingerprint — that is structural, not
  a convention. Every kind of step may name one; in practice only sources do, because a
  resource answers "where do these rows come from".
* *Code.* `model_class` and `resolver_class` are registry names, so the target
  environment must have the same classes registered (`add_model_class`). A document is
  portable across environments, not across codebases.
* *Labels.* Publishing is something you do to a *result*, using `Resolver.publish`. It
  is not part of the plan, so the receiving environment collects and then publishes
  under whatever label it wants. The only names in a document are sources' names,
  which are part of their output.
* *Data, or any hash of it.* A source's fingerprint folds in a content hash of what it
  actually read, and that hash is derived on load by reading the target's own rows.
  Same rows give the same fingerprints, so the target store hits cache instead of
  recomputing. Different rows give different fingerprints, so it re-runs. Both are the
  intended behaviour.

A rebuilt plan therefore fingerprints identically, as long as every methodology it names
declares a version. One that declares none is keyed by a nonce on every collect, here as
anywhere else, so the steps from it down re-run wherever the document is loaded. See
`matchlab.core.versioning`.
"""

from collections.abc import Mapping
from typing import Any, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from matchlab.core.exceptions import ResourceError
from matchlab.core.kinds import StepKind
from matchlab.lineage import walk
from matchlab.models import Model
from matchlab.recordstep import RecordStep
from matchlab.resolvers import Resolver
from matchlab.resources import Resource, names_of
from matchlab.sources import Source
from matchlab.specs import (
    ModelSpec,
    ResolverSpec,
    SourceSpec,
    TransformSpec,
)
from matchlab.steps import Step
from matchlab.transformers import Transform

StepSpec = SourceSpec | TransformSpec | ModelSpec | ResolverSpec

SpecT = TypeVar("SpecT", bound=BaseModel)
StepT = TypeVar("StepT", bound=Step)

_SPEC_TYPES: dict[StepKind, type[BaseModel]] = {
    StepKind.SOURCE: SourceSpec,
    StepKind.TRANSFORM: TransformSpec,
    StepKind.MODEL: ModelSpec,
    StepKind.RESOLVER: ResolverSpec,
}


class StepNode(BaseModel):
    """One step: its kind, what it specifies, what it reads, and what it needs."""

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
    resources: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Settings field name to resource name, for every field this step's "
            "methodology was given as a `matchlab.resources.Resource`. Here rather "
            "than in the spec because a resource describes reconstruction, not output: "
            "renaming one changes no fingerprint. `load` fills each from its "
            "`resources` argument."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _spec_from_kind(cls, data: Any) -> Any:  # noqa: ANN401 - raw pydantic input
        """Parse `spec` as the type `kind` names, rather than guessing.

        The specs are structurally close enough that a plain union mis-assigns. A
        `TransformSpec` and a `ModelSpec` overlap once optional settings default, so a
        union could bind the wrong one. `kind` is the discriminator the document holds.
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

    def required_resources(self) -> set[str]:
        """Every resource name `load` must be given to rebuild this plan.

        Read straight off each node, because a resource is named in its own field
        rather than hidden among settings. See `matchlab.resources`.
        """
        return {name for node in self.steps for name in node.resources.values()}


def dump(root: Step, *, indent: int | None = None) -> str:
    """Describe `root` and everything it reads as a portable document.

    JSON rather than an object, because a document exists to be put somewhere: a
    column, an object store, a request body, a file.

    Args:
        root: The apex of the plan. Only what is reachable upstream of it is included,
            exactly as `collect()` would run.
        indent: How the JSON is laid out. `None`, the default, is compact.

    Returns:
        The document, as JSON.

    Raises:
        ResourceError: If a resource was passed without a name, so there is nothing for
            a document to record. A name covering two different objects is refused
            earlier, when the plan is built.
    """
    ordered = walk(root)

    # Check resources
    for position, step in enumerate(ordered):
        for field, resource in step._resources().items():
            if resource.name is None:
                raise ResourceError(
                    f"Step {position} ({step.kind}) was given an unnamed resource for "
                    f"'{field}', so this plan cannot be described. Wrap it: "
                    f'{field}=Resource("some_name", ...).'
                )

    position = {id(step): index for index, step in enumerate(ordered)}
    document = PlanDocument(
        steps=tuple(
            _node(step, tuple(position[id(parent)] for parent in step.parents))
            for step in ordered
        )
    )
    return document.model_dump_json(indent=indent)


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
        resources=names_of(step._resources()),
    )


def load(document: str | bytes, resources: Mapping[str, Any]) -> Step:
    """Rebuild a plan from a document, returning its apex.

    Nothing is collected. This reconstructs the same lazy plan, so the returned step
    fingerprints identically to the one that was dumped, given the same data.

    Args:
        document: The JSON `dump` produced.
        resources: Name to object, for every resource the document names.

    Returns:
        The plan's apex, the last step in the document.

    Raises:
        ValidationError: If the JSON does not describe a plan document.
        ValueError: If the document is empty, names an unregistered location class, or
            wires a step to an input of the wrong kind.
        ResourceError: If a named resource was not supplied.
    """
    parsed = PlanDocument.model_validate_json(document)
    if not parsed.steps:
        raise ValueError("A plan document needs at least one step")

    _check_supplied(parsed, resources)

    built: list[Step] = []
    for position, node in enumerate(parsed.steps):
        built.append(
            _rebuild(node, position, [built[i] for i in node.inputs], resources)
        )
    return built[-1]


def _check_supplied(document: PlanDocument, supplied: Mapping[str, Any]) -> None:
    """Reject a load missing any resource the document names, naming them all.

    Raises:
        ResourceError: If the document names a resource `supplied` does not have.
    """
    missing: dict[str, str] = {}
    for position, node in enumerate(document.steps):
        for field, name in node.resources.items():
            if name not in supplied and name not in missing:
                missing[name] = f"step {position} ({node.kind}) reads it as '{field}'"
    if not missing:
        return

    wanted = "\n".join(f"  '{name}' — {where}" for name, where in missing.items())
    example = ", ".join(f"'{name}': ..." for name in missing)
    raise ResourceError(
        f"This plan needs resources that were not supplied:\n{wanted}\n"
        f"Pass them when loading: resources={{{example}}}."
    )


def _rebuild(
    node: StepNode, position: int, inputs: list[Step], supplied: Mapping[str, Any]
) -> Step:
    """Rebuild one step from its node and its already-rebuilt inputs."""
    resources = {
        field: Resource(name, supplied[name]) for field, name in node.resources.items()
    }
    match node.kind:
        case StepKind.SOURCE:
            spec = _expect(node, position, SourceSpec)
            return Source(
                location_class=spec.location_class,
                name=spec.name,
                location_settings=spec.location_settings,
                location_resources=resources,
                key_field=spec.key_field,
            )

        case StepKind.TRANSFORM:
            spec = _expect(node, position, TransformSpec)
            record_steps = _all_of(node, position, inputs, RecordStep)
            if len(record_steps) != 1:
                raise ValueError(
                    f"Step {position} (transform) reads {len(record_steps)} record "
                    "steps; a transform reshapes exactly one."
                )
            return Transform(
                record_steps[0],
                transformer=spec.transformer_class,
                transformer_settings=spec.transformer_settings,
                transformer_resources=resources,
            )

        case StepKind.MODEL:
            spec = _expect(node, position, ModelSpec)
            record_steps = _all_of(node, position, inputs, RecordStep)
            if not 1 <= len(record_steps) <= 2:
                raise ValueError(
                    f"Step {position} (model) reads {len(record_steps)} record "
                    "steps; a deduper reads one and a linker two."
                )
            return Model(
                left=record_steps[0],
                right=record_steps[1] if len(record_steps) > 1 else None,
                model_class=spec.model_class,
                model_settings=spec.model_settings,
                model_resources=resources,
            )

        case StepKind.RESOLVER:
            spec = _expect(node, position, ResolverSpec)
            return Resolver(
                *_all_of(node, position, inputs, Model),
                resolver_class=spec.resolver_class,
                resolver_settings=spec.resolver_settings,
                resolver_resources=resources,
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
