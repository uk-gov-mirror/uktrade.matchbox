"""Unit tests for the DuckDB storage adapter (Phase 1).

The adapter is storage-only: these tests exercise round-trips, schema validation,
garbage collection, on-disk persistence, and a real evaluation round-trip through
`matchlab.core.eval.precision_recall`.
"""

from pathlib import Path

import polars as pl
import pytest

from matchlab.adapters.duckdb import DuckDBAdapter
from matchlab.core.eval import Judgement, precision_recall
from matchlab.core.exceptions import SchemaMismatch


@pytest.fixture
def adapter() -> DuckDBAdapter:
    """An ephemeral in-memory adapter."""
    a = DuckDBAdapter(":memory:")
    yield a
    a.close()


# Distinct fingerprints for each artifact under test.
FP_SRC = b"\x01" * 32
FP_SRC_B = b"\x0b" * 32
FP_MODEL = b"\x02" * 32
FP_RESOLVER = b"\x03" * 32
FP_RESOLVER_B = b"\x0c" * 32


def _extract() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "company_name": ["acme", "acme ltd", "beta"],
            "postcode": ["AB1", "AB1", "CD2"],
            "key": ["k1", "k2", "k3"],
        }
    )


def _leaves() -> pl.DataFrame:
    return pl.DataFrame(
        {"key": ["k1", "k2", "k3"], "leaf": [1, 2, 3]},
        schema={"key": pl.Utf8, "leaf": pl.UInt64},
    )


def _edges() -> pl.DataFrame:
    return pl.DataFrame(
        {"left_id": [1, 3], "right_id": [2, 4], "score": [0.9, 0.8]},
        schema={"left_id": pl.UInt64, "right_id": pl.UInt64, "score": pl.Float32},
    )


def _resolution() -> pl.DataFrame:
    # root/leaf/key/source == SCHEMA_EVAL_SAMPLES
    return pl.DataFrame(
        {
            "root": [10, 10, 20, 20],
            "leaf": [1, 2, 3, 4],
            "key": ["k1", "k2", "k3", "k4"],
            "source": ["crn", "crn", "dh", "dh"],
        },
        schema={
            "root": pl.UInt64,
            "leaf": pl.UInt64,
            "key": pl.Utf8,
            "source": pl.Utf8,
        },
    )


# -- existence + source round-trip ----------------------------------------------------


def test_has_is_false_before_store(adapter: DuckDBAdapter) -> None:
    assert not adapter.has(FP_SRC)


def test_source_round_trip(adapter: DuckDBAdapter) -> None:
    adapter.store_source(FP_SRC, "key", _extract(), _leaves())

    assert adapter.has(FP_SRC)
    extract = adapter.read_source_extract(FP_SRC).sort("key")
    assert extract.equals(_extract().sort("key"))

    leaves = adapter.read_source_leaves(FP_SRC).sort("key")
    assert leaves.equals(_leaves().sort("key"))
    assert leaves.schema["leaf"] == pl.UInt64


def test_read_missing_source_raises(adapter: DuckDBAdapter) -> None:
    with pytest.raises(KeyError):
        adapter.read_source_extract(FP_SRC)


# -- model + resolver round-trips -----------------------------------------------------


def test_model_round_trip(adapter: DuckDBAdapter) -> None:
    adapter.store_model(FP_MODEL, _edges())
    out = adapter.read_model(FP_MODEL).sort("left_id")
    assert out.equals(_edges().sort("left_id"))


def test_resolver_round_trip(adapter: DuckDBAdapter) -> None:
    adapter.store_resolver(FP_RESOLVER, _resolution())
    out = adapter.read_resolver(FP_RESOLVER).sort("leaf")
    assert out.equals(_resolution().sort("leaf"))


def test_store_resolver_rejects_bad_schema(adapter: DuckDBAdapter) -> None:
    bad = pl.DataFrame({"root": [1], "leaf": [2]})  # missing key/source
    with pytest.raises(SchemaMismatch):
        adapter.store_resolver(FP_RESOLVER, bad)


def test_store_model_rejects_bad_schema(adapter: DuckDBAdapter) -> None:
    bad = pl.DataFrame({"left_id": [1], "right_id": [2]})  # missing score
    with pytest.raises(SchemaMismatch):
        adapter.store_model(FP_MODEL, bad)


# -- idempotency ----------------------------------------------------------------------


def test_store_is_idempotent(adapter: DuckDBAdapter) -> None:
    adapter.store_resolver(FP_RESOLVER, _resolution())
    adapter.store_resolver(FP_RESOLVER, _resolution())  # no duplicate rows
    assert adapter.read_resolver(FP_RESOLVER).height == _resolution().height


# -- sampling -------------------------------------------------------------------------


def test_sample_returns_whole_clusters_and_is_seed_stable(
    adapter: DuckDBAdapter,
) -> None:
    adapter.store_resolver(FP_RESOLVER, _resolution())

    sample_a = adapter.sample(FP_RESOLVER, n=1, seed=42)
    sample_b = adapter.sample(FP_RESOLVER, n=1, seed=42)
    assert sample_a.equals(sample_b)

    # Sampling one root returns all of that root's rows, nothing partial.
    roots = sample_a["root"].unique().to_list()
    assert len(roots) == 1
    full = _resolution().filter(pl.col("root") == roots[0]).sort("leaf")
    assert sample_a.sort("leaf").equals(full)


def test_sample_caps_at_available_clusters(adapter: DuckDBAdapter) -> None:
    adapter.store_resolver(FP_RESOLVER, _resolution())
    everything = adapter.sample(FP_RESOLVER, n=999)
    assert everything.height == _resolution().height


# -- evaluation round-trip (Phase 1 exit criterion) -----------------------------------


def test_eval_round_trip_scores_a_perfect_model(adapter: DuckDBAdapter) -> None:
    """Store a judgement, read it back, and score a model that matches it exactly."""
    # User shown leaves 1-4, endorses {1,2} and {3,4} as two separate entities.
    adapter.store_judgement(
        Judgement(shown=[1, 2, 3, 4], endorsed=[[1, 2], [3, 4]]), user_name="alice"
    )

    judgements, expansion = adapter.read_eval_data()
    assert judgements.height == 2  # one row per endorsed group

    # A model that clusters exactly {1,2} and {3,4}.
    model_root_leaf = pl.DataFrame(
        {"root": [100, 100, 200, 200], "leaf": [1, 2, 3, 4]},
        schema={"root": pl.UInt64, "leaf": pl.UInt64},
    )

    (precision, recall) = precision_recall([model_root_leaf], judgements, expansion)[0]
    assert precision == 1.0
    assert recall == 1.0


def test_eval_tag_filtering(adapter: DuckDBAdapter) -> None:
    adapter.store_judgement(Judgement(tag="v1", shown=[1, 2], endorsed=[[1, 2]]))
    adapter.store_judgement(Judgement(tag="v2", shown=[3, 4], endorsed=[[3, 4]]))

    assert adapter.read_eval_data(tag="v1")[0].height == 1
    assert adapter.read_eval_data()[0].height == 2


def test_read_eval_data_empty_has_correct_schema(adapter: DuckDBAdapter) -> None:
    judgements, expansion = adapter.read_eval_data()
    assert judgements.height == 0
    assert set(judgements.columns) == {"user_name", "endorsed", "shown"}
    assert set(expansion.columns) == {"root", "leaves"}


# -- garbage collection ---------------------------------------------------------------


def test_gc_removes_only_dead_artifacts(adapter: DuckDBAdapter) -> None:
    adapter.store_source(FP_SRC, "key", _extract(), _leaves())
    adapter.store_source(FP_SRC_B, "key", _extract(), _leaves())
    adapter.store_model(FP_MODEL, _edges())
    adapter.store_resolver(FP_RESOLVER, _resolution())

    removed = adapter.gc(live={FP_SRC, FP_RESOLVER})
    assert removed == 2  # FP_SRC_B and FP_MODEL

    assert adapter.has(FP_SRC)
    assert adapter.has(FP_RESOLVER)
    assert not adapter.has(FP_SRC_B)
    assert not adapter.has(FP_MODEL)

    # The dropped source's extract table is really gone, but the live one survives.
    with pytest.raises(KeyError):
        adapter.read_source_extract(FP_SRC_B)
    assert adapter.read_source_extract(FP_SRC).height == _extract().height


# -- on-disk persistence --------------------------------------------------------------


def test_persists_across_reopen(tmp_path: Path) -> None:
    db = tmp_path / "nested" / "store.duckdb"
    a = DuckDBAdapter(db)
    a.store_resolver(FP_RESOLVER, _resolution())
    a.close()

    b = DuckDBAdapter(db)
    try:
        assert b.has(FP_RESOLVER)
        assert b.read_resolver(FP_RESOLVER).height == _resolution().height
    finally:
        b.close()


def test_a_label_is_a_movable_pointer(adapter: DuckDBAdapter) -> None:
    """A label points at a fingerprint, and moves when told to.

    The adapter does not argue about overwriting — that is `Resolver.publish`'s call.
    What it guarantees is that a label resolves to exactly one fingerprint, which the
    old (kind, name) column on `artifacts` did not: several generations shared a name
    and the lookup picked whichever row came back first.
    """
    adapter.store_resolver(FP_RESOLVER, _resolution())
    adapter.store_resolver(FP_RESOLVER_B, _resolution())

    assert adapter.find("entities") is None
    assert adapter.labels() == []

    adapter.publish("entities", FP_RESOLVER)
    assert adapter.find("entities") == FP_RESOLVER
    assert adapter.labels() == ["entities"]

    adapter.publish("entities", FP_RESOLVER_B)
    assert adapter.find("entities") == FP_RESOLVER_B
    assert adapter.labels() == ["entities"]  # moved, not duplicated


def test_purging_an_artifact_drops_labels_pointing_at_it(
    adapter: DuckDBAdapter,
) -> None:
    """A label resolving to an artifact that no longer exists would be a trap."""
    adapter.store_resolver(FP_RESOLVER, _resolution())
    adapter.publish("entities", FP_RESOLVER)

    adapter.gc(live=set())
    assert adapter.find("entities") is None
    assert adapter.labels() == []
