"""Scenario builders — ready-made plans with known ground truth.

The previous version of this module was ~1,300 lines because building a scenario meant
driving a server: create a collection, create a run, create each step, upload its data,
mark the run default. None of that exists now. A scenario is just a plan, so a builder
is a handful of lines: generate linked sources with known entities, write them to a
warehouse, and wire up the nodes.

Every builder returns a `Scenario` carrying both the plan and the truth it was built
from, so a test can collect the plan and assert the result against the planted
entities.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import Engine, create_engine

from matchlab.core.factories.models import (
    ModelTestkit,
    ScriptedLinker,
    ScriptedLinkerSettings,
    model_factory,
    register_truth,
    truth_from_testkits,
)
from matchlab.core.factories.sources import (
    LinkedSourcesTestkit,
    SourceTestkit,
    linked_sources_factory,
)
from matchlab.models.models import Model
from matchlab.resolvers import Resolver


@dataclass
class Scenario:
    """A built plan plus the ground truth it was generated from."""

    linked: LinkedSourcesTestkit
    engine: Engine
    models: dict[str, ModelTestkit] = field(default_factory=dict)
    apex: Resolver | None = None

    @property
    def sources(self) -> dict[str, SourceTestkit]:
        """The generated source testkits, by name."""
        return self.linked.sources

    @property
    def true_entities(self) -> tuple:
        """The planted entities, as a tuple."""
        return tuple(self.linked.true_entities)


def _linked(n_true_entities: int, seed: int, engine: Engine | None) -> Scenario:
    """Generate linked sources and write them to a warehouse the plan can read."""
    engine = engine or create_engine("sqlite:///:memory:")
    linked = linked_sources_factory(
        n_true_entities=n_true_entities, engine=engine, seed=seed
    )
    for testkit in linked.sources.values():
        testkit.write_to_location(set_client=engine)
    return Scenario(linked=linked, engine=engine)


def source_scenario(
    n_true_entities: int = 10, seed: int = 42, engine: Engine | None = None
) -> Scenario:
    """Sources only — generated, written to the warehouse, no models."""
    return _linked(n_true_entities, seed, engine)


def dedupe_scenario(
    n_true_entities: int = 10, seed: int = 42, engine: Engine | None = None
) -> Scenario:
    """One deduping model over `crn`, resolved."""
    scenario = _linked(n_true_entities, seed, engine)

    model = model_factory(
        left_testkit=scenario.sources["crn"],
        true_entities=scenario.true_entities,
        seed=seed,
    )
    scenario.models["dedupe_crn"] = model
    scenario.apex = model.resolve()
    return scenario


def link_scenario(
    n_true_entities: int = 10, seed: int = 42, engine: Engine | None = None
) -> Scenario:
    """`crn` deduped, then linked to `cdms` *through* that resolver.

    This is the layered shape worth exercising: the link reads `crn` resolved by the
    upstream dedupe, so the apex must carry that grouping forward (merge-forward) as
    well as its own.
    """
    scenario = _linked(n_true_entities, seed, engine)
    crn, cdms = scenario.sources["crn"], scenario.sources["cdms"]

    dedupe = model_factory(
        left_testkit=crn,
        true_entities=scenario.true_entities,
        seed=seed,
    )
    resolved_crn = dedupe.resolve()

    truth_id = register_truth(
        truth_from_testkits(
            left_testkit=crn,
            source_entities=scenario.true_entities,
            right_testkit=cdms,
        )
    )
    link = Model(
        left=resolved_crn.view(crn.source),
        right=cdms.source.view(),
        model_class=ScriptedLinker,
        model_settings=ScriptedLinkerSettings(truth_id=truth_id),
    )

    scenario.models["dedupe_crn"] = dedupe
    scenario.apex = link.resolve()
    return scenario
