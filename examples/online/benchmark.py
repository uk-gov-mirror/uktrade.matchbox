"""matchlab vs plain Splink: expression, batch cost, and a single online match.

Both pipelines run the identical algorithm (dedupe each source, link every pair with the
shared Splink model, cluster once), so the comparison isolates what matchlab adds: a
plan, and a materialised store. We check they agree, time and memory-profile the batch
build, then time a single online lookup against matchlab's own DuckDB store versus the
same resolution in SQLite.

    python benchmark.py                # the whole story, one table
    python benchmark.py _one matchlab  # run one pipeline once (used by memray)
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
import timeit
import warnings
from pathlib import Path

import polars as pl

warnings.filterwarnings("ignore")
import logging  # noqa: E402

logging.disable(logging.WARNING)

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import data  # noqa: E402
import model  # noqa: E402
import pipeline_matchlab  # noqa: E402
import pipeline_splink  # noqa: E402

from matchlab.adapters import DuckDBAdapter  # noqa: E402

N_ROWS = 300_000
REPEATS = 3
N_PROBES = 2_000


def build_matchlab(trained: dict, on_disk: bool = False) -> None:
    """Collect the matchlab plan into a fresh store. Times/measures the build only.

    A fresh store each time, so content-addressed caching doesn't turn a repeat into a
    no-op. On disk, the materialised tables live in the DuckDB file rather than in RAM.
    """
    path = HERE / f".ml_{time.time_ns()}.duckdb" if on_disk else None
    store = DuckDBAdapter(path) if path else DuckDBAdapter()
    pipeline_matchlab.build_plan(trained).collect(store)
    store.close()
    if path:
        path.unlink(missing_ok=True)


def matchlab_resolution(trained: dict) -> pl.DataFrame:
    """The matchlab resolution, for the correctness check."""
    return pipeline_matchlab.build_plan(trained).collect(DuckDBAdapter()).resolution()


def _mixed(res: pl.DataFrame, truth: pl.DataFrame) -> int:
    """Clusters that mix more than one true entity (0 means perfect)."""
    return (
        res.join(truth, on=["source", "key"])
        .group_by("root")
        .agg(pl.col("entity").n_unique().alias("n"))
        .filter(pl.col("n") > 1)
        .height
    )


def _peak_mib(which: str) -> float:
    """Peak heap of one pipeline, measured by memray in a subprocess."""
    binf, jsonf = HERE / f".mem_{which}.bin", HERE / f".mem_{which}.json"
    run = ["uv", "run", "memray"]
    kw = {"check": True, "capture_output": True, "cwd": HERE}
    subprocess.run(
        [*run, "run", "-q", "-f", "-o", str(binf), __file__, "_one", which], **kw
    )
    subprocess.run([*run, "stats", "--json", "-fo", str(jsonf), str(binf)], **kw)
    peak = json.loads(jsonf.read_text())["metadata"]["peak_memory"]
    binf.unlink(), jsonf.unlink()
    return peak / 1024**2


def _time(lookup, probes: list) -> float:
    """Mean latency of a single lookup, in milliseconds."""
    lookup(*probes[0])  # warm up
    start = time.perf_counter()
    for source, key in probes:
        lookup(source, key)
    return (time.perf_counter() - start) / len(probes) * 1000


def _one(which: str) -> None:
    data.ensure_warehouse(N_ROWS)
    trained = model.load_or_train(pipeline_splink.all_nodes())
    if which == "splink":
        pipeline_splink.resolution(trained)
    else:
        build_matchlab(trained, on_disk=which == "matchlab_disk")


def main() -> None:
    data.ensure_warehouse(N_ROWS)
    truth = data.truth()
    trained = model.load_or_train(pipeline_splink.all_nodes())
    print(f"{truth.height:,} records, {truth['entity'].n_unique():,} true entities\n")

    # Expression: both pipelines must agree with the ground truth.
    for name, res in (
        ("matchlab", matchlab_resolution(trained)),
        ("splink", pipeline_splink.resolution(trained)),
    ):
        good = res["root"].n_unique() == truth["entity"].n_unique() and not _mixed(
            res, truth
        )
        print(f"expression       {name:<9} {'ok' if good else 'FAIL'}")
    print()

    # Batch build: time (fresh store each repeat) and peak memory. matchlab is timed
    # for the build only (the store is the product), both in memory and on disk.
    def best(fn):
        return min(timeit.repeat(fn, number=1, repeat=REPEATS))

    ts = best(lambda: pipeline_splink.resolution(trained))
    tm = best(lambda: build_matchlab(trained))
    td = best(lambda: build_matchlab(trained, on_disk=True))
    ms, mm, md = _peak_mib("splink"), _peak_mib("matchlab"), _peak_mib("matchlab_disk")
    print(
        f"batch build      {'Splink':>10}{'matchlab (mem)':>16}{'matchlab (disk)':>17}"
    )
    print(f"  time         {ts:>8.1f} s{tm:>14.1f} s{td:>15.1f} s")
    print(f"  peak memory  {ms:>6,.0f} MiB{mm:>12,.0f} MiB{md:>13,.0f} MiB\n")

    # A single online match, against matchlab's own DuckDB store and the same
    # resolution in SQLite. No extra copies: the store is the one the plan built.
    store_path = HERE / ".online_store.duckdb"
    store_path.unlink(missing_ok=True)
    store = DuckDBAdapter(store_path)
    apex = pipeline_matchlab.build_plan(trained)
    apex.collect(store)
    fp = apex._fp
    con = store.conn

    def duck(source, key):
        root = con.execute(
            "SELECT root FROM resolution WHERE fp=? AND source=? AND key=?",
            [fp, source, key],
        ).fetchone()[0]
        return con.execute(
            "SELECT source, key FROM resolution WHERE fp=? AND root=?", [fp, root]
        ).fetchall()

    res = con.execute("SELECT root, key, source FROM resolution WHERE fp=?", [fp]).pl()
    sqlite_path = HERE / ".online_store.sqlite"
    sqlite_path.unlink(missing_ok=True)
    scon = sqlite3.connect(sqlite_path)
    scon.execute("CREATE TABLE resolution (root TEXT, key TEXT, source TEXT)")
    scon.executemany(
        "INSERT INTO resolution VALUES (?,?,?)",
        res.select(pl.col("root").cast(pl.Utf8), "key", "source").iter_rows(),
    )
    scon.execute("CREATE INDEX i1 ON resolution (source, key)")
    scon.execute("CREATE INDEX i2 ON resolution (root)")

    def row(source, key):
        root = scon.execute(
            "SELECT root FROM resolution WHERE source=? AND key=?", [source, key]
        ).fetchone()[0]
        return scon.execute(
            "SELECT source, key FROM resolution WHERE root=?", [root]
        ).fetchall()

    probes = list(res.select("source", "key").sample(N_PROBES, seed=1).iter_rows())
    duck_ms, row_ms = _time(duck, probes), _time(row, probes)
    print("single online match      latency     throughput")
    print(f"  DuckDB (matchlab store) {duck_ms:7.3f} ms   {1000 / duck_ms:>8,.0f} /s")
    print(f"  SQLite (row store)      {row_ms:7.3f} ms   {1000 / row_ms:>8,.0f} /s")

    scon.close()
    store.close()
    store_path.unlink(missing_ok=True)
    sqlite_path.unlink(missing_ok=True)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "_one":
        _one(sys.argv[2])
    else:
        main()
