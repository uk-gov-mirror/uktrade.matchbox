"""The same pipeline as `with_matchlab.py`, hand-rolled with Polars.

Not a strawman: this is how you'd reasonably write it. The matching itself is the same
exact-match rule, so the difference between the two files is pipeline machinery, not
matching cleverness.

Run `warehouse.py` first. Both files print the same answer.
"""

import sqlite3
from itertools import combinations
from pathlib import Path

import polars as pl

DB = Path(__file__).parent / "warehouse.sqlite"

# Each source keeps its own column names, so the pipeline has to know them.
SOURCES = {
    "crn": ("company_number", "company_name", "registered_postcode"),
    "dh": ("dh_id", "name", "postcode"),
    "cdms": ("account_ref", "account_name", "post_code"),
    "hmrc": ("trader_id", "trader_name", "postcode"),
}


class DisjointSet:
    """Union-find, for turning pairwise edges into groups."""

    def __init__(self) -> None:
        self.parent: dict[int, int] = {}

    def add(self, item: int) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:  # path compression
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, left: int, right: int) -> None:
        self.add(left)
        self.add(right)
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root

    def groups(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for item in self.parent:
            out.setdefault(self.find(item), []).append(item)
        return out


def clean(frame: pl.DataFrame, name_col: str, postcode_col: str) -> pl.DataFrame:
    """Normalise a name and postcode into comparable form."""
    return frame.with_columns(
        pl.col(name_col)
        .str.to_uppercase()
        .str.replace_all(r"\s+|\.|\bLIMITED\b|\bLTD\b|\bPLC\b", "")
        .alias("name"),
        pl.col(postcode_col)
        .str.to_uppercase()
        .str.replace_all(r"\s+", "")
        .alias("postcode"),
    )


def load() -> pl.DataFrame:
    """Read every source into one frame with a global record ID.

    Sources have different key and field names, and a global ID space is needed
    because matches cross sources — a per-source row number isn't enough.
    """
    frames = []
    with sqlite3.connect(DB) as conn:
        for source, (key, name_col, postcode_col) in SOURCES.items():
            rows = conn.execute(
                f"select {key}, {name_col}, {postcode_col} from {source}"
            ).fetchall()
            frame = pl.DataFrame(
                rows, schema=[key, name_col, postcode_col], orient="row"
            )
            frames.append(
                clean(frame, name_col, postcode_col).select(
                    pl.lit(source).alias("source"),
                    pl.col(key).cast(pl.Utf8).alias("key"),
                    "name",
                    "postcode",
                )
            )

    records = pl.concat(frames)
    # Stable IDs, so a rerun on unchanged data gives the same answer.
    return records.sort("source", "key").with_row_index("record_id")


def exact_match_pairs(
    left: pl.DataFrame, right: pl.DataFrame, id_column: str
) -> list[tuple[int, int]]:
    """Every pair agreeing on cleaned name and postcode."""
    joined = left.join(right, on=["name", "postcode"], how="inner", suffix="_right")
    return [
        (row[id_column], row[f"{id_column}_right"])
        for row in joined.iter_rows(named=True)
        if row[id_column] != row[f"{id_column}_right"]
    ]


def resolve() -> dict[int, list[str]]:
    """Dedupe each source, link the deduplicated entities, return key groups."""
    records = load()

    # --- 1. Dedupe within each source ------------------------------------------
    dedupe = DisjointSet()
    for record_id in records["record_id"]:
        dedupe.add(record_id)  # so records nothing matches survive as singletons

    for source in SOURCES:
        rows = records.filter(pl.col("source") == source)
        for left, right in exact_match_pairs(rows, rows, "record_id"):
            dedupe.union(left, right)

    # Each record's entity after deduplication.
    entity_of = {
        record: root for root, members in dedupe.groups().items() for record in members
    }
    records = records.with_columns(
        pl.col("record_id")
        .replace_strict(entity_of, return_dtype=pl.UInt32)
        .alias("entity_id")
    )

    # --- 2. Project into entity space -------------------------------------------
    # Linking compares entities, not records, so each entity needs one row. Records
    # in an entity agree on name and postcode by construction, so `first` is safe
    # here — with fuzzier deduplication it would not be, and this is where you'd
    # have to decide what an entity's name *is*.
    entities = (
        records.group_by("source", "entity_id")
        .agg(pl.col("name").first(), pl.col("postcode").first())
        .sort("source", "entity_id")
    )

    # --- 3. Link every pair of sources ------------------------------------------
    link = DisjointSet()
    for entity_id in entities["entity_id"]:
        link.add(entity_id)

    for left_source, right_source in combinations(SOURCES, 2):
        left = entities.filter(pl.col("source") == left_source)
        right = entities.filter(pl.col("source") == right_source)
        for left_id, right_id in exact_match_pairs(left, right, "entity_id"):
            link.union(left_id, right_id)

    # --- 4. Back to records -----------------------------------------------------
    # The link merged *entities*; every record inside a merged entity has to come
    # with it, or deduplication is silently undone.
    final_of_entity = {
        entity: root for root, members in link.groups().items() for entity in members
    }
    grouped: dict[int, list[str]] = {}
    for row in records.iter_rows(named=True):
        final = final_of_entity.get(row["entity_id"], row["entity_id"])
        grouped.setdefault(final, []).append(row["key"])

    return grouped


if __name__ == "__main__":
    from warehouse import truth

    groups = resolve()
    total = sum(len(keys) for keys in groups.values())
    print(f"{len(groups)} entities from {total} records\n")

    expected = truth()
    for keys in sorted(groups.values(), key=lambda k: sorted(k)[0]):
        entities = {expected[key] for key in keys}
        mark = "✓" if len(entities) == 1 else "✗"
        print(f"  {mark} {sorted(keys)}")
