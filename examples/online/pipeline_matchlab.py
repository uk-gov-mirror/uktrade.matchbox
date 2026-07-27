"""The blog's pipeline as a matchlab plan.

Dedupe each dataset, link every pair (four sources, so six models, the blog's "small
number" middle ground), then consolidate all edges and cluster once. matchlab is an
opinionated implementation of exactly that shape: a DAG of content-addressed artefacts.
"""

from __future__ import annotations

from itertools import combinations

from data import SOURCES, WAREHOUSE
from model import CLEAN_NAME, CLEAN_POSTCODE, THRESHOLD, _settings
from sqlalchemy import create_engine

from matchlab import Source
from matchlab.locations import RelationalDBLocation
from matchlab.models.dedupers import NaiveDeduper
from matchlab.models.linkers import SplinkLinker
from matchlab.models.linkers.splinklinker import SplinkSettings


def build_plan(trained: dict, warehouse=WAREHOUSE):
    """Build the lazy plan. Nothing runs until it is collected."""
    location = RelationalDBLocation(name="warehouse").set_client(
        create_engine(f"sqlite:///{warehouse}")
    )
    link_settings = SplinkSettings(
        left_id="id",
        right_id="id",
        threshold=THRESHOLD,
        linker_settings=_settings("link_only", trained),  # reuse the artefact
        linker_training_functions=[],  # no per-link training
    )

    sources, nodes = {}, {}
    for name, (key_col, name_col, postcode_col) in SOURCES.items():
        src = Source(
            location=location,
            name=name,
            extract_transform=(
                f"select {key_col}, {name_col}, {postcode_col} from {name}"
            ),
            key_field=key_col,
        )
        sources[name] = src
        clean_name = CLEAN_NAME.format(src.f(name_col))
        clean_postcode = CLEAN_POSTCODE.format(src.f(postcode_col))
        # Dedupe each source on its standardised name and postcode.
        deduped = src.view(cleaning={"name": clean_name, "postcode": clean_postcode})
        nodes[name] = deduped

    dedupes = [
        v.dedupe(NaiveDeduper, {"unique_fields": ["name", "postcode"]})
        for v in nodes.values()
    ]
    resolver = dedupes[0].resolve(*dedupes[1:])

    # Standardised entity nodes: one row per entity, read through the dedupe.
    entity_nodes = {}
    for name, src in sources.items():
        _, name_col, postcode_col = SOURCES[name]
        entity_nodes[name] = resolver.view(
            src,
            cleaning={
                "name": f"any_value({CLEAN_NAME.format(src.f(name_col))})",
                "postcode": f"any_value({CLEAN_POSTCODE.format(src.f(postcode_col))})",
            },
            group=True,
        )

    # Link every pair (Splink), then consolidate all edges and cluster once.
    links = [
        entity_nodes[a].link(entity_nodes[b], SplinkLinker, link_settings)
        for a, b in combinations(entity_nodes, 2)
    ]
    return links[0].resolve(*links[1:])
