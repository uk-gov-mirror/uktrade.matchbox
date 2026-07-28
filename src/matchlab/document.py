"""Serialise a plan to a portable document, and rebuild it somewhere else.

A `PlanDocument` is a **derived view** of a plan, not a source format: humans write
Python, and this is what you hand to another environment so it can execute the same
plan. So it is dumped, transferred and loaded — never hand-authored, which is why
nodes refer to each other by position rather than by any human-facing name.

Nodes come out in `lineage.walk` order, so every input index is smaller than the index
of the step that consumes it. Positions also preserve structural sharing: a view
feeding two models is one node referenced twice, where nesting each step's inputs
inside it would have inlined the whole subtree twice over.

**Edges live here, settings live in the config.** That split is the point of this
module. A config is hashed into a fingerprint, so it must carry everything that
changes the step's output and nothing else; a serialised plan must additionally carry
the edges, which a fingerprint already covers by folding in its parents'. Putting both
in one model meant either a config that lied about identity — a rename invalidating a
subtree that produces identical bytes — or a document that could not be rebuilt.

Edges are not the only thing that lands on this side of the split. A source's
`LocationRef` says which location class to build and under what name, so that `load`
can attach a client — and neither fact changes a byte the source produces, which is why
a `SourceConfig` records nothing about where its rows came from.

**What a document cannot carry:**

* *Clients.* A document names the locations its sources read and stops there. `load`
  takes the clients and attaches them by name, so credentials never enter a document.
* *Code.* `model_class` and `resolver_class` are registry names, so the target
  environment must have the same classes registered (`add_model_class`). A document is
  portable across environments, not across codebases.
* *Labels.* Publishing is something you do to a *result* — `Resolver.publish` — not
  part of the plan, so the receiving environment collects and then publishes under
  whatever label it wants. The only names in a document are sources', which are part
  of their output.
* *Data, or any hash of it.* A source's fingerprint folds in a content hash of what it
  actually read, and that is derived on load from the target warehouse. Same rows,
  same fingerprints, and the target store hits cache instead of recomputing; different
  rows, different fingerprints, and it re-runs. Both are the intended behaviour.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from matchlab.core.config import (
    ModelConfig,
    ResolverConfig,
    SourceConfig,
    ViewConfig,
)
from matchlab.lineage import walk
from matchlab.locations import build_location
from matchlab.models import Model
from matchlab.resolvers import Resolver
from matchlab.sources import Source
from matchlab.views import View

if TYPE_CHECKING:
    from collections.abc import Mapping

    from matchlab.steps import Step

StepKind = Literal["source", "view", "model", "resolver"]
StepConfig = SourceConfig | ViewConfig | ModelConfig | ResolverConfig

ConfigT = TypeVar("ConfigT", bound=BaseModel)
StepT = TypeVar("StepT", bound="Step")

_CONFIG_TYPES: dict[str, type[BaseModel]] = {
    "source": SourceConfig,
    "view": ViewConfig,
    "model": ModelConfig,
    "resolver": ResolverConfig,
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
    """One step: what kind it is, how it is configured, and what it reads."""

    model_config = ConfigDict(frozen=True)

    kind: StepKind = Field(description="Which kind of step this is.")
    config: StepConfig = Field(
        description="This step's own settings — exactly what its fingerprint hashes."
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
            "the config because it describes reconstruction rather than output — see "
            "`LocationRef`. Every other kind of step leaves it unset."
        ),
    )

    @model_validator(mode="after")
    def _check_location(self) -> StepNode:
        """A source names a location; nothing else has one to name."""
        if self.kind == "source" and self.location is None:
            raise ValueError("A source step must name the location it reads.")
        if self.kind != "source" and self.location is not None:
            raise ValueError(f"A {self.kind} step must not name a location.")
        return self

    @model_validator(mode="before")
    @classmethod
    def _config_from_kind(cls, data: Any) -> Any:  # noqa: ANN401 - raw pydantic input
        """Parse `config` as the type `kind` names, rather than guessing.

        The four configs are structurally close enough that a plain union mis-assigns:
        `ViewConfig` has no required fields, so an empty object matches it and several
        others. `kind` is the discriminator the document already carries.
        """
        if not isinstance(data, dict):
            return data
        kind, config = data.get("kind"), data.get("config")
        if isinstance(config, dict) and kind in _CONFIG_TYPES:
            return {**data, "config": _CONFIG_TYPES[kind].model_validate(config)}
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
    def _check_edges(self) -> PlanDocument:
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
    """Describe one step, checking its config is the one its kind implies."""
    config_type = _CONFIG_TYPES.get(step.kind)
    if config_type is None or not isinstance(step.config, config_type):
        raise ValueError(
            f"A {step.kind} step has no matching config type. A new kind of step "
            "needs an entry in `_CONFIG_TYPES`."
        )
    return StepNode(
        kind=cast(StepKind, step.kind),
        config=cast(StepConfig, step.config),
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

    Nothing is collected: this reconstructs the same lazy plan, so the returned step
    fingerprints identically to the one that was dumped, given the same data.

    Args:
        document: A dumped plan.
        clients: Location name to client, for every location the document's sources
            name. A `RelationalDBLocation` takes a SQLAlchemy engine or an ADBC
            connection.

    Returns:
        The plan's apex — the last step in the document.

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
        case "source":
            config = _expect(node, position, SourceConfig)
            reference = cast(LocationRef, node.location)
            if reference.name not in clients:
                raise ValueError(
                    f"Source '{config.name}' reads location '{reference.name}', which "
                    f"has no client. Pass one in `clients`: "
                    f"{{'{reference.name}': engine}}."
                )
            return Source(
                location=build_location(
                    reference.location_class,
                    name=reference.name,
                    client=clients[reference.name],
                ),
                name=config.name,
                extract_transform=config.extract_transform,
                key_field=config.key_field,
            )

        case "view":
            config = _expect(node, position, ViewConfig)
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
                cleaning=config.cleaning,
                group=config.group,
            )

        case "model":
            config = _expect(node, position, ModelConfig)
            views = _all_of(node, position, inputs, View)
            if not 1 <= len(views) <= 2:
                raise ValueError(
                    f"Step {position} (model) reads {len(views)} views; a deduper "
                    "reads one and a linker two."
                )
            return Model(
                left=views[0],
                right=views[1] if len(views) > 1 else None,
                model_class=config.model_class,
                model_settings=config.model_settings,
            )

        case "resolver":
            config = _expect(node, position, ResolverConfig)
            return Resolver(
                *_all_of(node, position, inputs, Model),
                resolver_class=config.resolver_class,
                resolver_settings=config.resolver_settings,
            )


def _expect(node: StepNode, position: int, config_type: type[ConfigT]) -> ConfigT:
    """Narrow a node's config to the type its kind implies."""
    if not isinstance(node.config, config_type):
        raise ValueError(  # pragma: no cover - the validator parses config by kind
            f"Step {position} ({node.kind}) has a "
            f"{type(node.config).__name__}, expected {config_type.__name__}."
        )
    return node.config


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
