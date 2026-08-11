"""Dummy linked-source pipeline for `just eval` to review.

Generates fresh testkit data on every run, so nothing is checked in.
"""

from matchlab.resolvers import Resolver
from matchlab.testkit import linked_sources_factory


def entities() -> Resolver:
    """Build a perfect-linker resolver over generated `crn`/`dh` sources."""
    linked = linked_sources_factory(n_true_entities=20).write_to_location()
    return linked.link("crn", "dh").model.resolve()
