"""The warehouse both pipelines read.

Reuses the fast vectorised generator in ``examples/companies/benchmark.py`` (the
factory subsystem's per-row Python generation is minutes-slow at this size). ~100k
true entities become ~390k records across four sources, each keeping its own column
names. A hidden ``_truth`` table records the real entity behind every key, for scoring.

    python data.py 300_000
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import polars as pl
from sqlalchemy import create_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "companies"))
import benchmark as companies  # noqa: E402

WAREHOUSE = Path(__file__).parent / "warehouse.sqlite"
# name -> (key column, name column, postcode column), as the companies example uses.
SOURCES: dict[str, tuple[str, str, str]] = dict(companies.COLUMNS)


def build_warehouse(n_rows: int = 300_000, path: Path = WAREHOUSE) -> Path:
    """Generate and write the warehouse: one table per source, plus hidden truth."""
    frames = companies.generate(n_rows)
    path.unlink(missing_ok=True)
    with sqlite3.connect(path) as conn:
        truth_rows: list[tuple[str, str, int]] = []
        for source, (key_col, name_col, postcode_col) in SOURCES.items():
            frame = frames[source]
            conn.execute(
                f"CREATE TABLE {source} "
                f"({key_col} TEXT, {name_col} TEXT, {postcode_col} TEXT)"
            )
            conn.executemany(
                f"INSERT INTO {source} VALUES (?, ?, ?)",
                frame.select("key", "name", "postcode").iter_rows(),
            )
            conn.execute(f"CREATE INDEX idx_{source}_key ON {source} ({key_col})")
            truth_rows.extend(
                (source, key, entity)
                for key, entity in frame.select("key", "_truth").iter_rows()
            )
        conn.execute("CREATE TABLE _truth (source TEXT, key TEXT, entity INTEGER)")
        conn.executemany("INSERT INTO _truth VALUES (?, ?, ?)", truth_rows)
    return path


def ensure_warehouse(n_rows: int = 300_000, path: Path = WAREHOUSE) -> Path:
    """Build the warehouse if missing, otherwise reuse it."""
    if not path.exists():
        build_warehouse(n_rows, path)
    return path


def truth(path: Path = WAREHOUSE) -> pl.DataFrame:
    """The hidden ground truth `(source, key, entity)`, for scoring only."""
    engine = create_engine(f"sqlite:///{path}")
    return pl.read_database("SELECT source, key, entity FROM _truth", engine.connect())


if __name__ == "__main__":
    n = int(sys.argv[1].replace("_", "")) if len(sys.argv) > 1 else 300_000
    t = truth(build_warehouse(n))
    print(f"warehouse: {WAREHOUSE}")
    print(f"  {t.height:,} keys, {t['entity'].n_unique():,} true entities")
