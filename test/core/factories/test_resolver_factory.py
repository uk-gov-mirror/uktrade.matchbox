"""Tests for the resolver testkit factory.

Ported to the plan API. The tests that exercised the removed machinery — `into_dag`,
`fake_run`, and DAG detachment — are gone with it: models are plan nodes now, so there
is no DAG to attach to or detach from, and a resolver's inputs are its `upstream`.
"""

import pytest

from matchlab.core.factories.models import model_factory
from matchlab.core.factories.resolvers import (
    MockResolver,
    ResolverTestkit,
    resolver_factory,
)
from matchlab.core.factories.sources import linked_sources_factory
from matchlab.resolvers import Resolver


def test_resolver_factory_can_autobuild() -> None:
    """The factory stands on its own, generating its own model input."""
    testkit = resolver_factory()

    assert isinstance(testkit, ResolverTestkit)
    assert isinstance(testkit.resolver, Resolver)
    assert testkit.resolver.resolver_class is MockResolver
    assert not testkit.resolver.is_collected

    assert len(testkit.resolver.inputs) == 1
    # Thresholds key by input position
    assert testkit.resolver.resolver_settings.thresholds == {0: 0.0}


def test_resolver_inputs_become_plan_upstream() -> None:
    """A resolver's model inputs are its upstream references."""
    linked = linked_sources_factory()
    crn = model_factory(
        name="dedupe_crn",
        left_testkit=linked.sources["crn"],
        true_entities=tuple(linked.true_entities),
    )
    dh = model_factory(
        name="dedupe_dh",
        left_testkit=linked.sources["dh"],
        true_entities=tuple(linked.true_entities),
    )

    testkit = resolver_factory(inputs=[crn, dh])

    # Steps have no names, so identity is the comparison — which is also how the
    # plan itself deduplicates nodes.
    assert set(testkit.resolver.upstream) == {crn.model, dh.model}
    # Both models' sources are reachable through the resolver's lineage.
    sources = {
        step.name for step in testkit.resolver.lineage() if hasattr(step, "name")
    }
    assert {"crn", "dh"} <= sources


def test_resolver_factory_requires_testkit_inputs() -> None:
    """Only ModelTestkits are accepted as inputs."""
    linked = linked_sources_factory()
    crn = model_factory(
        left_testkit=linked.sources["crn"],
        true_entities=tuple(linked.true_entities),
    )
    resolver_inner = resolver_factory(inputs=[crn])

    with pytest.raises(TypeError, match="resolver_factory inputs must be ModelTestkit"):
        resolver_factory(inputs=[crn.model])  # a plan node, not a testkit

    with pytest.raises(TypeError, match="resolver_factory inputs must be ModelTestkit"):
        resolver_factory(inputs=[resolver_inner, crn])


def test_resolver_factory_honours_explicit_thresholds() -> None:
    """Thresholds gate which edges contribute to the expected assignments."""
    linked = linked_sources_factory()
    model_testkit = model_factory(
        left_testkit=linked.sources["crn"],
        true_entities=tuple(linked.true_entities),
        score_range=(0.5, 0.99),
    )
    assert model_testkit.scores.height > 0

    low = resolver_factory(inputs=[model_testkit], thresholds={model_testkit.name: 0.0})
    high = resolver_factory(
        inputs=[model_testkit], thresholds={model_testkit.name: 1.0}
    )

    assert low.assignments.height > 0
    assert high.assignments.height == 0


def test_resolver_factory_rejects_mismatched_thresholds() -> None:
    """Threshold keys must name exactly the resolver's inputs."""
    linked = linked_sources_factory()
    crn = model_factory(
        left_testkit=linked.sources["crn"],
        true_entities=tuple(linked.true_entities),
    )

    with pytest.raises(ValueError, match="Threshold keys must exactly match"):
        resolver_factory(inputs=[crn], thresholds={"not_a_model": 0.0})
