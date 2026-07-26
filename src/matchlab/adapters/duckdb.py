"""DuckDB storage adapter — the reference local backend for matchlab.

A single DuckDB database (a file, or `:memory:`) holds every collected artifact,
keyed by step fingerprint. There is no resolution engine here: resolvers arrive
already materialised (merge-forward), and reads are plain table scans. Analysts can
point their own SQL at the `resolution` table — it is the whole point.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import duckdb
import polars as pl

from matchlab.adapters.base import Adapter, Fingerprint
from matchlab.core.arrow import (
    SCHEMA_CLUSTER_EXPANSION,
    SCHEMA_EVAL_SAMPLES,
    SCHEMA_JUDGEMENTS,
    SCHEMA_MODEL_EDGES,
    check_schema_subset,
)
from matchlab.core.eval import Judgement
from matchlab.core.resolution import root_id_of

#: Bumped whenever the stored shape changes. A store written by an older matchlab is
#: recreated rather than half-read, which is the honest failure for a cache.
_SCHEMA_VERSION = 3

_SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS meta (
    schema_version INTEGER
);
CREATE TABLE IF NOT EXISTS artifacts (
    fp BLOB PRIMARY KEY, kind VARCHAR
);
-- A label: a pointer from a string someone chose to the resolution they want to find
-- again. Separate from `artifacts` because labelling is an act, not a property — most
-- artifacts carry none, and a label can be moved to a newer fingerprint without
-- disturbing the artifact it used to point at.
CREATE TABLE IF NOT EXISTS labels (
    label VARCHAR PRIMARY KEY, fp BLOB, published_at TIMESTAMP
);
-- A stored source knows its own key column, so its extract can be read back and
-- joined to a resolution without the plan that built it.
CREATE TABLE IF NOT EXISTS source_meta (
    fp BLOB PRIMARY KEY, key_field VARCHAR
);
-- Which source artifacts a resolution was built from. A resolution records source
-- *names*, and one store can hold several generations of the same name.
CREATE TABLE IF NOT EXISTS resolution_sources (
    fp BLOB, source_name VARCHAR, source_fp BLOB
);
CREATE TABLE IF NOT EXISTS source_leaves (
    fp BLOB, key VARCHAR, leaf UBIGINT
);
CREATE TABLE IF NOT EXISTS model_edges (
    fp BLOB, left_id UBIGINT, right_id UBIGINT, score FLOAT
);
CREATE TABLE IF NOT EXISTS resolution (
    fp BLOB, root UBIGINT, leaf UBIGINT, key VARCHAR, source VARCHAR
);
CREATE TABLE IF NOT EXISTS judgements (
    tag VARCHAR, user_name VARCHAR, endorsed UBIGINT, shown UBIGINT
);
CREATE TABLE IF NOT EXISTS expansion (
    root UBIGINT PRIMARY KEY, leaves UBIGINT[]
);
"""


def _mint_cluster_id(leaves: list[int]) -> int:
    """Deterministic, content-addressed cluster ID for a group of leaves.

    A singleton maps to the leaf itself, so evaluation's singleton-expansion fallback
    (`process_judgements`) works without an explicit expansion row.

    A group uses `root_id`, the same function a resolver mints its roots with. That
    is load-bearing rather than tidy: scoring compares a judged group against the
    resolution's clusters by ID, so if the two ever disagree every comparison misses
    and precision/recall is computed over an empty set.
    """
    if len(leaves) == 1:
        return int(leaves[0])
    return root_id_of(leaves)


class DuckDBAdapter(Adapter):
    """Store collected artifacts in a DuckDB database, keyed by fingerprint."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        """Open (or create) the store at `path`. Use `:memory:` for ephemeral stores."""
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(self.path)
        self._open_schema()

    def _open_schema(self) -> None:
        """Create the schema, recreating the store if it predates this version."""
        existing = self.conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_name = 'meta'"
        ).fetchone()
        if existing:
            row = self.conn.execute("SELECT schema_version FROM meta").fetchone()
            if row and row[0] == _SCHEMA_VERSION:
                return

        # Either a pre-versioned store or an older version: start clean. Artifacts are
        # a cache — everything in here can be recomputed from the plan that made it.
        for (table,) in self.conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'main'"
        ).fetchall():
            self.conn.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')

        self.conn.execute(_SCHEMA_DDL)
        self.conn.execute("INSERT INTO meta VALUES (?)", [_SCHEMA_VERSION])

    # -- helpers ----------------------------------------------------------------------

    @staticmethod
    def _extract_table(fp: Fingerprint) -> str:
        """Name of the per-source extract table (arbitrary-schema, so its own table)."""
        return f"extract_{fp.hex()}"

    @staticmethod
    def _clean_table(fp: Fingerprint) -> str:
        """Name of the per-view cleaned table (arbitrary-schema, so its own table)."""
        return f"clean_{fp.hex()}"

    def _register(self, name: str, df: pl.DataFrame) -> None:
        self.conn.register(name, df.to_arrow())

    def _kind(self, fp: Fingerprint) -> str | None:
        row = self.conn.execute(
            "SELECT kind FROM artifacts WHERE fp = ?", [fp]
        ).fetchone()
        return row[0] if row else None

    def _register_artifact(self, fp: Fingerprint, kind: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO artifacts VALUES (?, ?)", [fp, kind])

    def _purge(self, fp: Fingerprint) -> None:
        """Remove any existing artifact for `fp`, so stores are idempotent.

        Only ever called immediately before storing that same fingerprint again. A
        fingerprint addresses content, so the replacement is the same data by
        construction — which is why any **label** pointing at `fp` is left alone. The
        label still resolves, to bytes indistinguishable from the ones it resolved to
        before, and a publication is not something a re-collect should quietly revoke.
        """
        kind = self._kind(fp)
        if kind is None:
            return
        if kind == "source":
            self.conn.execute(f'DROP TABLE IF EXISTS "{self._extract_table(fp)}"')
            self.conn.execute("DELETE FROM source_leaves WHERE fp = ?", [fp])
            self.conn.execute("DELETE FROM source_meta WHERE fp = ?", [fp])
        elif kind == "clean":
            self.conn.execute(f'DROP TABLE IF EXISTS "{self._clean_table(fp)}"')
        elif kind == "model":
            self.conn.execute("DELETE FROM model_edges WHERE fp = ?", [fp])
        elif kind == "resolver":
            self.conn.execute("DELETE FROM resolution WHERE fp = ?", [fp])
            self.conn.execute("DELETE FROM resolution_sources WHERE fp = ?", [fp])
        self.conn.execute("DELETE FROM artifacts WHERE fp = ?", [fp])

    # -- existence --------------------------------------------------------------------

    def has(self, fp: Fingerprint) -> bool:  # noqa: D102
        return self._kind(fp) is not None

    # -- sources ----------------------------------------------------------------------

    def store_source(  # noqa: D102
        self,
        fp: Fingerprint,
        key_field: str,
        extract: pl.DataFrame,
        leaves: pl.DataFrame,
    ) -> None:
        missing = {"key", "leaf"} - set(leaves.columns)
        if missing:
            raise ValueError(f"leaves is missing columns: {sorted(missing)}")

        self._purge(fp)

        self._register("_reg_extract", extract)
        self.conn.execute(
            f'CREATE OR REPLACE TABLE "{self._extract_table(fp)}" '
            "AS SELECT * FROM _reg_extract"
        )
        self.conn.unregister("_reg_extract")

        leaves = leaves.select(
            pl.col("key").cast(pl.Utf8), pl.col("leaf").cast(pl.UInt64)
        )
        self._register("_reg_leaves", leaves)
        self.conn.execute(
            "INSERT INTO source_leaves SELECT ? AS fp, key, leaf FROM _reg_leaves", [fp]
        )
        self.conn.unregister("_reg_leaves")

        self.conn.execute("INSERT INTO source_meta VALUES (?, ?)", [fp, key_field])
        self._register_artifact(fp, "source")

    def read_source_extract(self, fp: Fingerprint) -> pl.DataFrame:  # noqa: D102
        if self._kind(fp) != "source":
            raise KeyError(f"No stored source for fingerprint {fp.hex()}")
        return self.conn.execute(f'SELECT * FROM "{self._extract_table(fp)}"').pl()

    def read_source_leaves(self, fp: Fingerprint) -> pl.DataFrame:  # noqa: D102
        if self._kind(fp) != "source":
            raise KeyError(f"No stored source for fingerprint {fp.hex()}")
        return self.conn.execute(
            "SELECT key, leaf FROM source_leaves WHERE fp = ?", [fp]
        ).pl()

    # -- models -----------------------------------------------------------------------

    def store_model(self, fp: Fingerprint, edges: pl.DataFrame) -> None:  # noqa: D102
        check_schema_subset(SCHEMA_MODEL_EDGES, edges.to_arrow().schema)
        self._purge(fp)
        edges = edges.select(
            pl.col("left_id").cast(pl.UInt64),
            pl.col("right_id").cast(pl.UInt64),
            pl.col("score").cast(pl.Float32),
        )
        self._register("_reg_edges", edges)
        self.conn.execute(
            "INSERT INTO model_edges "
            "SELECT ? AS fp, left_id, right_id, score FROM _reg_edges",
            [fp],
        )
        self.conn.unregister("_reg_edges")
        self._register_artifact(fp, "model")

    def read_model(self, fp: Fingerprint) -> pl.DataFrame:  # noqa: D102
        if self._kind(fp) != "model":
            raise KeyError(f"No stored model for fingerprint {fp.hex()}")
        return self.conn.execute(
            "SELECT left_id, right_id, score FROM model_edges WHERE fp = ?", [fp]
        ).pl()

    # -- cleaned views ----------------------------------------------------------------

    def store_clean(self, fp: Fingerprint, table: pl.DataFrame) -> None:  # noqa: D102
        self._purge(fp)
        self._register("_reg_clean", table)
        self.conn.execute(
            f'CREATE OR REPLACE TABLE "{self._clean_table(fp)}" '
            "AS SELECT * FROM _reg_clean"
        )
        self.conn.unregister("_reg_clean")
        self._register_artifact(fp, "clean")

    def read_clean(self, fp: Fingerprint) -> pl.DataFrame:  # noqa: D102
        if self._kind(fp) != "clean":
            raise KeyError(f"No stored cleaned view for fingerprint {fp.hex()}")
        return self.conn.execute(f'SELECT * FROM "{self._clean_table(fp)}"').pl()

    # -- resolvers --------------------------------------------------------------------

    def store_resolver(  # noqa: D102
        self,
        fp: Fingerprint,
        resolution: pl.DataFrame,
        sources: Mapping[str, Fingerprint] | None = None,
    ) -> None:
        check_schema_subset(SCHEMA_EVAL_SAMPLES, resolution.to_arrow().schema)
        self._purge(fp)
        resolution = resolution.select(
            pl.col("root").cast(pl.UInt64),
            pl.col("leaf").cast(pl.UInt64),
            pl.col("key").cast(pl.Utf8),
            pl.col("source").cast(pl.Utf8),
        )
        self._register("_reg_res", resolution)
        self.conn.execute(
            "INSERT INTO resolution "
            "SELECT ? AS fp, root, leaf, key, source FROM _reg_res",
            [fp],
        )
        self.conn.unregister("_reg_res")
        for source_name, source_fp in (sources or {}).items():
            self.conn.execute(
                "INSERT INTO resolution_sources VALUES (?, ?, ?)",
                [fp, source_name, source_fp],
            )
        self._register_artifact(fp, "resolver")

    def read_resolver(self, fp: Fingerprint) -> pl.DataFrame:  # noqa: D102
        if self._kind(fp) != "resolver":
            raise KeyError(f"No stored resolver for fingerprint {fp.hex()}")
        return self.conn.execute(
            "SELECT root, leaf, key, source FROM resolution WHERE fp = ?", [fp]
        ).pl()

    # -- evaluation -------------------------------------------------------------------

    def store_judgement(  # noqa: D102
        self, judgement: Judgement, user_name: str = "local"
    ) -> None:
        shown_id = _mint_cluster_id(judgement.shown)
        self._upsert_expansion(shown_id, judgement.shown)

        for group in judgement.endorsed:
            endorsed_id = _mint_cluster_id(group)
            self._upsert_expansion(endorsed_id, group)
            self.conn.execute(
                "INSERT INTO judgements VALUES (?, ?, ?, ?)",
                [judgement.tag, user_name, endorsed_id, shown_id],
            )

    def _upsert_expansion(self, root: int, leaves: list[int]) -> None:
        self.conn.execute(
            "INSERT INTO expansion (root, leaves) "
            "SELECT ?, CAST(? AS UBIGINT[]) ON CONFLICT (root) DO NOTHING",
            [root, sorted(int(leaf) for leaf in leaves)],
        )

    def read_eval_data(  # noqa: D102
        self, tag: str | None = None
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        if tag is None:
            judgements = self.conn.execute(
                "SELECT user_name, endorsed, shown FROM judgements"
            ).pl()
        else:
            judgements = self.conn.execute(
                "SELECT user_name, endorsed, shown FROM judgements WHERE tag = ?",
                [tag],
            ).pl()
        expansion = self.conn.execute("SELECT root, leaves FROM expansion").pl()

        # Present empty results with the right columns/dtypes. We deliberately do NOT
        # re-validate against the arrow transport schemas here: those pin `leaves` to a
        # small `list`, whereas polars naturally emits `large_list` — a serialisation
        # detail that is meaningless locally. Types are guaranteed by the table DDL and
        # inputs are validated on write.
        if judgements.height == 0:
            judgements = pl.DataFrame(schema=pl.Schema(SCHEMA_JUDGEMENTS))
        if expansion.height == 0:
            expansion = pl.DataFrame(schema=pl.Schema(SCHEMA_CLUSTER_EXPANSION))

        return judgements, expansion

    def sample(  # noqa: D102
        self, resolver_fp: Fingerprint, n: int, seed: int | None = None
    ) -> pl.DataFrame:
        resolution = self.read_resolver(resolver_fp)
        roots = resolution["root"].unique()
        if n < roots.len():
            roots = roots.sample(n=n, seed=seed, shuffle=True)
        return resolution.filter(pl.col("root").is_in(roots.to_list())).select(
            "root", "leaf", "key", "source"
        )

    # -- lookups ----------------------------------------------------------------------

    def publish(self, label: str, fp: Fingerprint) -> None:  # noqa: D102
        self.conn.execute(
            "INSERT OR REPLACE INTO labels VALUES (?, ?, now())", [label, fp]
        )

    def find(self, label: str) -> Fingerprint | None:  # noqa: D102
        row = self.conn.execute(
            "SELECT fp FROM labels WHERE label = ?", [label]
        ).fetchone()
        return row[0] if row else None

    def labels(self) -> list[str]:  # noqa: D102
        return [
            row[0]
            for row in self.conn.execute(
                "SELECT label FROM labels ORDER BY label"
            ).fetchall()
        ]

    def source_key_field(self, fp: Fingerprint) -> str:  # noqa: D102
        row = self.conn.execute(
            "SELECT key_field FROM source_meta WHERE fp = ?", [fp]
        ).fetchone()
        if row is None:
            raise KeyError(f"No stored source for fingerprint {fp.hex()}")
        return row[0]

    def resolution_sources(self, fp: Fingerprint) -> dict[str, Fingerprint]:  # noqa: D102
        return {
            name: source_fp
            for name, source_fp in self.conn.execute(
                "SELECT source_name, source_fp FROM resolution_sources WHERE fp = ?",
                [fp],
            ).fetchall()
        }

    # -- lifecycle --------------------------------------------------------------------

    def close(self) -> None:  # noqa: D102
        self.conn.close()
