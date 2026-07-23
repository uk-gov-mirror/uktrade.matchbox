"""End-to-end local DAG run — no server, no factories (Phase 2).

Builds a real DAG over a SQLite warehouse and a DuckDB adapter, runs it, and queries
it. Exercises the whole rewired pipeline: source hashing + leaf assignment + storage,
querying a bare source and querying *through* an upstream resolver, deduping, linking,
`materialise_resolution`, and the terminal `get_matches` / `lookup_key` reads.

The scenario deliberately covers the merge-forward property end to end:

    crn: a1=(acme,london) a2=(acme,leeds) a3=(beta,hull)
    dh:  b1=(acme,bristol) b2=(gamma,york)
    dedupe(crn) groups {a1, a2} on company; link(deduped-crn, dh) joins acme.

a1 and a2 differ on town, so they are distinct leaves — the deduper does real work
(rather than the rows collapsing at source-indexing). The apex is a *link* resolver,
yet the dedupe grouping of {a1, a2} must survive through it (fall-through), and a3 / b2
— reachable but touched by no edge — must stay singletons.
"""

from pathlib import Path

import polars as pl
import pytest
from sqlalchemy import Engine, create_engine, text

from matchbox.adapters import DuckDBAdapter
from matchbox.client.dags import DAG
from matchbox.client.locations import RelationalDBLocation
from matchbox.client.models.dedupers import NaiveDeduper
from matchbox.client.models.linkers import DeterministicLinker
from matchbox.client.resolvers.components import Components


@pytest.fixture
def warehouse(tmp_path: Path) -> Engine:
    """A SQLite warehouse with two overlapping company tables."""
    engine = create_engine(f"sqlite:///{tmp_path / 'warehouse.sqlite'}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE crn (pk TEXT, company TEXT, town TEXT)"))
        conn.execute(
            text(
                "INSERT INTO crn VALUES "
                "('a1', 'acme', 'london'), "
                "('a2', 'acme', 'leeds'), "
                "('a3', 'beta', 'hull')"
            )
        )
        conn.execute(text("CREATE TABLE dh (pk TEXT, company TEXT, town TEXT)"))
        conn.execute(
            text(
                "INSERT INTO dh VALUES "
                "('b1', 'acme', 'bristol'), "
                "('b2', 'gamma', 'york')"
            )
        )
    return engine


def _build_dag(warehouse: Engine) -> DAG:
    dag = DAG("companies", adapter=DuckDBAdapter(":memory:"))

    location = RelationalDBLocation(name="warehouse")
    location.set_client(warehouse)

    crn = dag.source(
        location=location,
        name="crn",
        extract_transform="select pk, company, town from crn",
        key_field="pk",
        index_fields=["company", "town"],
        infer_types=True,
    )
    dh = dag.source(
        location=location,
        name="dh",
        extract_transform="select pk, company, town from dh",
        key_field="pk",
        index_fields=["company", "town"],
        infer_types=True,
    )

    # Dedupe crn on company, then wrap it in a resolver so it can be queried through.
    d_crn = crn.query().deduper(
        name="d_crn",
        model_class=NaiveDeduper,
        model_settings={"unique_fields": [crn.f("company")]},
    )
    r_crn = d_crn.resolver(name="r_crn", resolver_class=Components)

    # Link deduped-crn to dh on company; wrap in the apex resolver.
    crn_dh = crn.query(resolver=r_crn).linker(
        dh.query(),
        name="crn_dh",
        model_class=DeterministicLinker,
        model_settings={"comparisons": f"l.{crn.f('company')} = r.{dh.f('company')}"},
    )
    crn_dh.resolver(name="apex", resolver_class=Components)

    return dag


def test_local_dag_run_query_and_lookup(warehouse: Engine) -> None:
    dag = _build_dag(warehouse)
    dag.run_and_sync()

    lookup = dag.get_matches(resolver="apex").as_lookup()

    # crn_pk -> matchbox id and dh_pk -> matchbox id
    crn_ids = {
        row["crn_pk"]: row["id"]
        for row in lookup.filter(pl.col("crn_pk").is_not_null()).iter_rows(named=True)
    }
    dh_ids = {
        row["dh_pk"]: row["id"]
        for row in lookup.filter(pl.col("dh_pk").is_not_null()).iter_rows(named=True)
    }

    # Dedupe survived through the apex link: a1 and a2 share a cluster.
    assert crn_ids["a1"] == crn_ids["a2"]
    # Link joined acme across sources: b1 is in the same cluster.
    assert dh_ids["b1"] == crn_ids["a1"]
    # Fall-through: untouched records stay in their own clusters.
    assert crn_ids["a3"] != crn_ids["a1"]
    assert dh_ids["b2"] != crn_ids["a1"]
    assert crn_ids["a3"] != dh_ids["b2"]


def test_lookup_key_crosses_sources(warehouse: Engine) -> None:
    dag = _build_dag(warehouse)
    dag.run_and_sync()

    result = dag.lookup_key(from_source="crn", to_sources=["dh"], key="a1")

    assert set(result["crn"]) == {"a1", "a2"}
    assert set(result["dh"]) == {"b1"}


def test_rerun_is_cache_stable(warehouse: Engine) -> None:
    """Re-running an unchanged DAG hits the adapter cache and yields the same result."""
    dag = _build_dag(warehouse)
    dag.run_and_sync()
    first = dag.lookup_key(from_source="crn", to_sources=["dh"], key="a1")

    dag.run_and_sync()  # fingerprints unchanged -> stores skipped
    second = dag.lookup_key(from_source="crn", to_sources=["dh"], key="a1")

    assert first == second
