"""Unit tests for the DuckDB storage adapter (Phase 1).

The adapter is storage-only: these tests exercise round-trips, schema validation,
introspection, trimming, on-disk persistence, and a real evaluation round-trip through
`matchlab.core.eval.precision_recall`.
"""

from pathlib import Path

import polars as pl
import pytest

from matchlab.adapters import (
    DuckDBAdapter,
    DuckDBStoreStats,
    StoreStats,
    format_bytes,
)
from matchlab.core.exceptions import SchemaMismatch
from matchlab.core.kinds import StepKind
from matchlab.eval.judgements import Judgement
from matchlab.eval.metrics import precision_recall


@pytest.fixture
def adapter() -> DuckDBAdapter:
    """An ephemeral in-memory adapter."""
    a = DuckDBAdapter(":memory:")
    yield a
    a.close()


# Distinct fingerprints for each artifact under test.
FP_SRC = b"\x01" * 32
FP_MODEL = b"\x02" * 32
FP_RESOLVER = b"\x03" * 32
FP_RESOLVER_B = b"\x0c" * 32
FP_VIEW = b"\x04" * 32


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
    # root/leaf/key/source == SCHEMA_RESOLUTION
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


# -- introspection --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("count", "signed", "expected"),
    [
        (0, False, "0 B"),
        (0, True, "+0 B"),
        (512, False, "512 B"),
        (1024, False, "1.0 KB"),
        (1536, True, "+1.5 KB"),
        (5 * 1024**3, False, "5.0 GB"),
        (-2048, True, "-2.0 KB"),  # a store only shrinks if something rewrites it
    ],
)
def test_byte_formatting(count: int, signed: bool, expected: str) -> None:
    assert format_bytes(count, signed=signed) == expected


def test_stats_describe_themselves() -> None:
    stats = StoreStats(location="somewhere", bytes=2048, artifacts={StepKind.VIEW: 3})

    assert stats.describe() == "Store 2.0 KB, 3 artifacts"
    assert (
        stats.describe(since=StoreStats(location="somewhere", bytes=1024))
        == "Store 2.0 KB (+1.0 KB), 3 artifacts"
    )


def test_an_empty_store_describes_itself_without_a_plural() -> None:
    one = StoreStats(location="x", bytes=0, artifacts={StepKind.SOURCE: 1})

    assert StoreStats(location="x", bytes=0).describe() == "Store 0 B, 0 artifacts"
    assert one.describe() == "Store 0 B, 1 artifact"


def test_a_duckdb_store_says_when_its_bytes_are_only_resident() -> None:
    """The subclass hook: `4.6 MB` reads as disk, and for `:memory:` it is not."""
    resident = DuckDBStoreStats(location=":memory:", bytes=4096)
    on_disk = DuckDBStoreStats(location="/s.duckdb", bytes=4096, path=Path("/s.duckdb"))

    assert resident.describe() == "Store 4.0 KB in memory, 0 artifacts"
    assert on_disk.describe() == "Store 4.0 KB, 0 artifacts"


def test_stats_counts_what_is_stored(adapter: DuckDBAdapter) -> None:
    assert adapter.stats().artifacts == {}

    adapter.store_source(FP_SRC, "key", _extract(), _leaves())
    adapter.store_model(FP_MODEL, _edges())
    adapter.store_resolver(FP_RESOLVER, _resolution())
    adapter.publish("production", FP_RESOLVER)

    stats = adapter.stats()
    assert stats.artifacts == {"source": 1, "model": 1, "resolver": 1}
    assert stats.labels == 1


def test_an_in_memory_store_reports_a_real_size(adapter: DuckDBAdapter) -> None:
    """The regression guard for the in-memory branch.

    An in-memory store allocates no blocks, so anything reading DuckDB's block count
    reports `0 B` — for every test in this suite and every `DuckDBAdapter(":memory:")`
    a user writes. It has no file either, so there is nothing to `stat`.
    """
    adapter.store_source(FP_SRC, "key", _extract(), _leaves())

    stats = adapter.stats()

    assert stats.path is None
    assert stats.bytes > 0
    assert stats.free_bytes == 0


def test_stats_size_matches_what_is_on_disk(tmp_path: Path) -> None:
    """The figure has to survive the user checking it with `du`.

    Taken while the connection is open, which is when a collect reports: writes sit in
    the write-ahead log until they are settled into blocks, and a store measured before
    that reads orders of magnitude low.
    """
    db = tmp_path / "store.duckdb"
    store = DuckDBAdapter(db)
    try:
        store.store_resolver(FP_RESOLVER, _resolution())
        stats = store.stats()
        on_disk = sum(f.stat().st_size for f in tmp_path.iterdir())

        assert stats.path == db.resolve()
        assert stats.bytes == on_disk
    finally:
        store.close()

    # And it does not move once the store is closed and reopened.
    assert sum(f.stat().st_size for f in tmp_path.iterdir()) == stats.bytes


def test_stats_size_grows_with_what_is_stored(tmp_path: Path) -> None:
    store = DuckDBAdapter(tmp_path / "store.duckdb")
    try:
        empty = store.stats().bytes
        store.store_view(
            FP_VIEW, pl.DataFrame({"id": range(50_000), "name": ["x"] * 50_000})
        )
        assert store.stats().bytes > empty
    finally:
        store.close()


# -- trim -----------------------------------------------------------------------------


def _labelled_store(store: DuckDBAdapter) -> None:
    """A published resolution over one source, plus an unrelated model to throw away."""
    store.store_source(FP_SRC, "key", _extract(), _leaves())
    store.store_resolver(FP_RESOLVER, _resolution(), sources={"crn": FP_SRC})
    store.store_model(FP_MODEL, _edges())
    store.publish("entities", FP_RESOLVER)


def test_trim_keeps_what_it_was_told_and_drops_the_rest(tmp_path: Path) -> None:
    store = DuckDBAdapter(tmp_path / "store.duckdb")
    try:
        store.store_source(FP_SRC, "key", _extract(), _leaves())
        store.store_model(FP_MODEL, _edges())

        result = store.trim(keep=[FP_SRC])

        assert result.removed == 1
        assert result.kept == 1
        assert store.has(FP_SRC)
        assert not store.has(FP_MODEL)
        # The kept artifact is not merely listed — it still reads back.
        assert store.read_source_extract(FP_SRC).height == _extract().height
    finally:
        store.close()


def test_trim_actually_returns_space_to_the_disk(tmp_path: Path) -> None:
    """The assertion the old `gc()` would have failed.

    Purging alone frees nothing: DuckDB reuses freed blocks but never hands them back,
    so the file stays at its high-water mark until it is rewritten.
    """
    store = DuckDBAdapter(tmp_path / "store.duckdb")
    try:
        store.store_source(FP_SRC, "key", _extract(), _leaves())
        store.store_view(
            FP_VIEW, pl.DataFrame({"id": range(200_000), "name": ["padding"] * 200_000})
        )
        before = store.stats().bytes

        result = store.trim(keep=[FP_SRC])

        assert store.stats().bytes < before
        assert result.reclaimed > 0
    finally:
        store.close()


def test_trim_keeps_every_label_and_the_sources_it_reads_through(
    tmp_path: Path,
) -> None:
    """A publication survives a trim that never mentioned it — and stays *usable*.

    Keeping the label row alone is not enough. Reading a published resolution without a
    plan goes through `resolution_sources` to each source's extract, so a label kept
    without its sources resolves to a fingerprint whose data has gone, and fails with a
    bare `KeyError` well away from the cause.
    """
    store = DuckDBAdapter(tmp_path / "store.duckdb")
    try:
        _labelled_store(store)

        result = store.trim(keep=[])  # names nothing at all

        assert result.removed == 1  # the model, and only the model
        assert store.labels() == ["entities"]
        # Walk the whole label-only read path.
        fp = store.find("entities")
        assert fp == FP_RESOLVER
        assert store.sample(fp, n=10).height > 0
        assert store.resolution_sources(fp) == {"crn": FP_SRC}
        assert store.read_source_extract(FP_SRC).height == _extract().height
        assert store.source_key_field(FP_SRC) == "key"
    finally:
        store.close()


def test_trim_never_deletes_judgements(tmp_path: Path) -> None:
    """Human work, and the only thing in the store that cannot be recomputed."""
    store = DuckDBAdapter(tmp_path / "store.duckdb")
    try:
        _labelled_store(store)
        store.store_judgement(Judgement(shown=[1, 2, 3, 4], endorsed=[[1, 2], [3, 4]]))

        store.trim(keep=[])

        judgements, expansion = store.read_eval_data()
        assert judgements.height == 2
        assert expansion.height > 0
    finally:
        store.close()


def test_trimming_an_in_memory_store_does_not_empty_it(adapter: DuckDBAdapter) -> None:
    """An in-memory store must not be rewritten.

    Reopening `:memory:` hands back an empty database rather than a smaller one, so a
    close-and-swap would destroy the store it was asked to tidy — and `:memory:` is what
    the whole suite and both examples run on.
    """
    adapter.store_source(FP_SRC, "key", _extract(), _leaves())
    adapter.store_model(FP_MODEL, _edges())

    result = adapter.trim(keep=[FP_SRC])

    assert result.removed == 1
    assert adapter.has(FP_SRC)
    assert adapter.read_source_extract(FP_SRC).height == _extract().height


def test_trimming_with_nothing_to_keep_refuses(adapter: DuckDBAdapter) -> None:
    """An accidentally-empty list should not be how a store gets emptied."""
    adapter.store_source(FP_SRC, "key", _extract(), _leaves())

    with pytest.raises(ValueError, match="would empty the store"):
        adapter.trim(keep=[])

    assert adapter.has(FP_SRC)


# -- identifiers ----------------------------------------------------------------------


def test_identifiers_read_a_source_directly(adapter: DuckDBAdapter) -> None:
    """Without a resolver, a record's `id` is its own leaf."""
    adapter.store_source(FP_SRC, "key", _extract(), _leaves())

    out = adapter.read_identifiers(FP_SRC, "crn").sort("key")

    assert out.columns == ["id", "source", "key", "leaf"]
    assert out["id"].to_list() == [1, 2, 3]
    assert out["leaf"].to_list() == [1, 2, 3]
    assert out["source"].to_list() == ["crn"] * 3
    assert out.schema["id"] == pl.UInt64
    assert out.schema["source"] == pl.Utf8


def test_identifiers_read_through_a_resolver(adapter: DuckDBAdapter) -> None:
    """Through a resolver, `id` is the root cluster and `leaf` still names the row."""
    adapter.store_source(FP_SRC, "key", _extract(), _leaves())
    adapter.store_resolver(FP_RESOLVER, _resolution())

    out = adapter.read_identifiers(FP_SRC, "crn", FP_RESOLVER).sort("key")

    assert out.columns == ["id", "source", "key", "leaf"]
    # k1/k2 were clustered together upstream, so they share an id but keep their leaves.
    assert out["id"].to_list() == [10, 10]
    assert out["leaf"].to_list() == [1, 2]


def test_identifiers_filter_to_the_source_asked_for(adapter: DuckDBAdapter) -> None:
    """The whole point: one source's rows, not every source in the resolution.

    `_resolution()` holds crn and dh together. Reading crn must not return dh's rows —
    a plan linking every pair of sources asks for this once per pair, so returning the
    whole table and filtering afterwards is what made it quadratic.
    """
    adapter.store_source(FP_SRC, "key", _extract(), _leaves())
    adapter.store_resolver(FP_RESOLVER, _resolution())

    assert adapter.read_identifiers(FP_SRC, "crn", FP_RESOLVER).height == 2
    assert adapter.read_identifiers(FP_SRC, "dh", FP_RESOLVER).height == 2
    assert adapter.read_identifiers(FP_SRC, "nope", FP_RESOLVER).height == 0


def test_identifiers_reject_a_missing_artifact(adapter: DuckDBAdapter) -> None:
    with pytest.raises(KeyError):
        adapter.read_identifiers(FP_SRC, "crn")

    adapter.store_source(FP_SRC, "key", _extract(), _leaves())
    with pytest.raises(KeyError):
        adapter.read_identifiers(FP_SRC, "crn", FP_RESOLVER)


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


# -- on-disk persistence --------------------------------------------------------------


def test_persists_across_reopen(tmp_path: Path) -> None:
    db = tmp_path / "nested" / "store.duckdb"
    a = DuckDBAdapter(db)
    a.store_resolver(FP_RESOLVER, _resolution())
    a.publish("entities", FP_RESOLVER)
    a.store_judgement(Judgement(shown=[1, 2], endorsed=[[1, 2]]))
    a.close()

    b = DuckDBAdapter(db)
    try:
        assert b.has(FP_RESOLVER)
        assert b.read_resolver(FP_RESOLVER).height == _resolution().height
        # Publications and judgements survive a reopen too. Artifacts are a cache and
        # can be recomputed; these cannot, and `_open_schema` drops every table in the
        # database when the schema version moves — so this is what would catch a bump
        # taken without thinking about what else is in there.
        assert b.labels() == ["entities"]
        assert b.read_eval_data()[0].height == 1
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


def test_restoring_an_artifact_keeps_the_label_pointing_at_it(
    adapter: DuckDBAdapter,
) -> None:
    """Re-collecting a published plan must not quietly revoke the publication.

    Storing replaces the artifact for a fingerprint, and a fingerprint addresses
    content — so the label resolves to the same bytes it always did. Dropping it here
    used to mean a re-collect silently unpublished your resolution.
    """
    adapter.store_resolver(FP_RESOLVER, _resolution())
    adapter.publish("entities", FP_RESOLVER)

    adapter.store_resolver(FP_RESOLVER, _resolution())  # same fingerprint, again

    assert adapter.find("entities") == FP_RESOLVER
    assert adapter.labels() == ["entities"]
    assert adapter.read_resolver(FP_RESOLVER).height == _resolution().height
