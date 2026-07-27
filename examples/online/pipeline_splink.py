"""The same pipeline, hand-rolled with Splink instead of matchlab.

The fair opponent, in the spirit of `examples/companies/by_hand.py`: the *same*
algorithm the matchlab plan runs, written out by hand. Dedupe each source on the
standardised name and postcode, link every pair of deduped sources with the shared
Splink model, consolidate the edges, and cluster once with connected components. This is
also the blog's production shape: dedupe in isolation, pairwise transient links, cluster
once.

The only difference from `pipeline_matchlab.py` is that this holds intermediates in
memory and returns a frame, where matchlab builds a plan and materialises a store.
"""

from __future__ import annotations

import warnings
from itertools import combinations

import duckdb
import polars as pl
from data import SOURCES, WAREHOUSE
from model import CLEAN_NAME, CLEAN_POSTCODE, THRESHOLD, _settings
from splink import DuckDBAPI, Linker
from sqlalchemy import create_engine


def load_nodes(warehouse=WAREHOUSE) -> dict[str, pl.DataFrame]:
    """Load and standardise every source into `(id, name, postcode)` record frames."""
    engine = create_engine(f"sqlite:///{warehouse}")
    con = duckdb.connect(":memory:")
    nodes: dict[str, pl.DataFrame] = {}
    for source, (key_col, name_col, postcode_col) in SOURCES.items():
        raw = pl.read_database(
            f"select {key_col} as key, {name_col} as nm, {postcode_col} as pc "
            f"from {source}",
            engine.connect(),
        )
        con.register("raw", raw)
        nodes[source] = con.execute(
            f"select key as id, {CLEAN_NAME.format('nm')} as name, "
            f"{CLEAN_POSTCODE.format('pc')} as postcode from raw"
        ).pl()
        con.unregister("raw")
    con.close()
    return nodes


def all_nodes(warehouse=WAREHOUSE) -> pl.DataFrame:
    """Every source's standardised records in one frame, the model's training input."""
    return pl.concat(load_nodes(warehouse).values(), how="vertical")


def _dedupe(
    nodes: dict[str, pl.DataFrame],
) -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
    """Exact-dedupe each source on (name, postcode). Same rule as the NaiveDeduper.

    Returns one entity frame per source (`id`, name, postcode, one row per entity, id
    globally unique) and a `(source, key, id)` map from record key to its entity.
    """
    entities: dict[str, pl.DataFrame] = {}
    key_to_entity: list[pl.DataFrame] = []
    for source, df in nodes.items():
        grouped = df.group_by("name", "postcode").agg(pl.col("id").alias("keys"))
        grouped = grouped.with_row_index("i").with_columns(
            (pl.lit(f"{source}:") + pl.col("i").cast(pl.Utf8)).alias("id")
        )
        entities[source] = grouped.select("id", "name", "postcode")
        key_to_entity.append(
            grouped.explode("keys").select(
                pl.lit(source).alias("source"),
                pl.col("keys").alias("key"),
                "id",
            )
        )
    return entities, pl.concat(key_to_entity)


def _link(entities: dict[str, pl.DataFrame], trained: dict) -> list[tuple[str, str]]:
    """Link every pair of sources with the shared Splink model; return entity edges."""
    edges: list[tuple[str, str]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for left, right in combinations(entities, 2):
            linker = Linker(
                [entities[left].to_pandas(), entities[right].to_pandas()],
                _settings("link_only", trained),
                input_table_aliases=[left, right],
                db_api=DuckDBAPI(),
            )
            pred = linker.inference.predict(
                threshold_match_probability=THRESHOLD
            ).as_pandas_dataframe()
            edges.extend(zip(pred["id_l"], pred["id_r"], strict=True))
    return edges


def resolution(trained: dict, warehouse=WAREHOUSE) -> pl.DataFrame:
    """Dedupe, link, cluster once; return `(source, key, root)`."""
    entities, key_to_entity = _dedupe(load_nodes(warehouse))
    edges = _link(entities, trained)

    # Connected components over the entity edges (union-find), then key -> root.
    parent: dict[str, str] = {
        e: e for df in entities.values() for e in df["id"].to_list()
    }

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for left, right in edges:
        parent[find(left)] = find(right)

    roots = pl.DataFrame({"id": list(parent), "root": [find(e) for e in parent]})
    return key_to_entity.join(roots, on="id").select("source", "key", "root")
