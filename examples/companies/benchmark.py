"""Time both pipelines on data big enough to notice.

The 21-record example is for reading, not for timing: at that size everything is
sub-second and the only thing measured is fixed overhead. This generates the same
shape of data at whatever scale you ask for.

    python benchmark.py 10_000 100_000 1_000_000

Rows are split across the four sources, with duplicates inside each and overlap
between them, so both pipelines do real deduplication and real linking.
"""

import logging
import sqlite3
import sys
import time
from pathlib import Path

import polars as pl

logging.basicConfig(level=logging.INFO)

# The two pipelines sit alongside this file, so make them importable however this is
# run — as a script, or from the repository root.
sys.path.insert(0, str(Path(__file__).parent))

import by_hand  # noqa: E402
import with_matchlab  # noqa: E402

from matchlab.adapters import DuckDBAdapter  # noqa: E402

DB = Path(__file__).parent / "benchmark.sqlite"

SUFFIXES = ["LTD", "LIMITED", "PLC", "", "Ltd.", "ltd"]
TOWNS = ["LONDON", "MANCHESTER", "LEEDS", "BRISTOL", "EDINBURGH", "GLASGOW"]


def generate(n_rows: int, seed: int = 42) -> dict[str, pl.DataFrame]:
    """Build four sources totalling roughly `n_rows` records.

    Each company appears in one to four sources, and one to three times within a
    source with cosmetic variation — so deduplication and linking both have work to
    do, in roughly the proportions real data has.
    """
    rng = pl.Series("seed", [seed])  # noqa: F841 - documents intent; polars seeds below
    # ~2.6 rows per company on average across all sources.
    n_entities = max(1, n_rows // 3)

    base = pl.DataFrame(
        {
            "entity": pl.arange(0, n_entities, eager=True),
        }
    ).with_columns(
        pl.format("Company {}", pl.col("entity")).alias("stem"),
        (pl.col("entity") % len(TOWNS)).alias("town_idx"),
        # Deterministic pseudo-randomness: cheap, reproducible, no RNG plumbing.
        ((pl.col("entity") * 2654435761) % 1000).alias("hash"),
    )

    frames: dict[str, pl.DataFrame] = {}
    for index, source in enumerate(("crn", "dh", "cdms", "hmrc")):
        # Which companies this source knows about, and how many rows each gets.
        present = base.filter((pl.col("hash") + index * 250) % 1000 < 650)
        duplicated = present.with_columns(
            (1 + (pl.col("hash") + index) % 2).alias("copies")
        )
        rows = duplicated.select(
            pl.col("entity"),
            pl.col("stem"),
            pl.col("town_idx"),
            pl.col("hash"),
            pl.int_ranges(0, pl.col("copies")).alias("copy"),
        ).explode("copy")  # noqa: PD010

        frames[source] = rows.select(
            pl.format(
                "{}-{}-{}", pl.lit(source), pl.col("entity"), pl.col("copy")
            ).alias("key"),
            # Cosmetic variation the cleaning rules are meant to absorb.
            pl.format(
                "{} {}",
                pl.when((pl.col("hash") + pl.col("copy")) % 2 == 0)
                .then(pl.col("stem").str.to_uppercase())
                .otherwise(pl.col("stem")),
                pl.col("copy").map_elements(
                    lambda i: SUFFIXES[i % len(SUFFIXES)], return_dtype=pl.Utf8
                ),
            ).alias("name"),
            pl.format(
                "{}{} {}AA",
                pl.col("town_idx").map_elements(
                    lambda i: TOWNS[i][:2], return_dtype=pl.Utf8
                ),
                pl.col("entity") % 90 + 1,
                pl.col("entity") % 9 + 1,
            ).alias("postcode"),
            pl.col("entity").alias("_truth"),
        )

    return frames


# The example's column names, so both pipelines read this warehouse unchanged.
COLUMNS = {
    "crn": ("company_number", "company_name", "registered_postcode"),
    "dh": ("dh_id", "name", "postcode"),
    "cdms": ("account_ref", "account_name", "post_code"),
    "hmrc": ("trader_id", "trader_name", "postcode"),
}


def build(n_rows: int) -> int:
    """Write the benchmark warehouse. Returns the row count actually generated."""
    DB.unlink(missing_ok=True)
    frames = generate(n_rows)
    total = 0
    with sqlite3.connect(DB) as conn:
        for source, frame in frames.items():
            key, name, postcode = COLUMNS[source]
            conn.execute(
                f"CREATE TABLE {source} "
                f"({key} TEXT, {name} TEXT, {postcode} TEXT, _truth INTEGER)"
            )
            conn.executemany(
                f"INSERT INTO {source} VALUES (?, ?, ?, ?)", frame.iter_rows()
            )
            conn.execute(f"CREATE INDEX idx_{source} ON {source} ({key})")
            total += frame.height
    return total


def time_pipelines(n_rows: int) -> None:
    """Build data at this size and time both implementations over it."""

    actual = build(n_rows)
    by_hand.DB = DB
    with_matchlab.DB = DB

    start = time.perf_counter()
    groups_by_hand = by_hand.resolve()
    hand = time.perf_counter() - start

    store = DuckDBAdapter(":memory:")
    start = time.perf_counter()
    resolution = with_matchlab.build().collect(store).resolution()
    lab = time.perf_counter() - start

    start = time.perf_counter()
    with_matchlab.build().collect(store)
    cached = time.perf_counter() - start
    store.close()

    entities_lab = resolution["root"].n_unique()
    print(  # noqa: T201
        f"{actual:>10,} rows | by hand {hand:7.2f}s | matchlab {lab:7.2f}s "
        f"| re-collect {cached:6.2f}s | entities {entities_lab:,} "
        f"vs {len(groups_by_hand):,}"
    )


if __name__ == "__main__":
    sizes = [int(arg.replace("_", "")) for arg in sys.argv[1:]] or [10_000]
    print(f"{'rows':>10} | {'by hand':>15} | {'matchlab':>16} | cached | entities")  # noqa: T201
    for size in sizes:
        time_pipelines(size)
