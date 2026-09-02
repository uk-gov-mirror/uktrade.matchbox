"""The benchmark suite has to be measuring something real.

A benchmark that has quietly stopped resolving correctly still produces a beautiful
table — faster every release, and meaningless. These tests are the guard against that,
and they are cheap enough to run in CI: a few thousand rows, in process, no subprocess.

They cover the three things that would make a published number a lie: that the
generated data is actually hard to match, that each plan shape resolves what its shape
says it should, and that a measurement survives the JSON round-trip the runner puts it
through.
"""

import io
import json
from pathlib import Path

import polars as pl
import pytest
from rich.console import Console

from benchmarks.cases import BASELINE, Case, Store, profile
from benchmarks.data import VARIATIONS, Warehouse, ensure, generate
from benchmarks.measure import measure
from benchmarks.measurements import (
    Environment,
    Measurement,
    Result,
    Run,
    SweepResult,
)
from benchmarks.report import MIN_WIDTH, save, summarise
from benchmarks.topologies import Topology, build, declare
from matchlab.stores import DuckDBStore

SMALL = Warehouse(n_entities=300, records_per_entity=2, n_sources=4)

# A plausible environment, for the tests that need a `Run` but not a real one.
ENVIRONMENT = Environment(
    started_at="2026-01-01T00-00-00",
    profile="quick",
    repeats=1,
    store=Store.FILE,
    matchlab_version="0.0.0",
    git_sha="abc1234",
    python="3.13.0",
    platform="Darwin arm64",
    cpus=8,
    startup_s=0.4,
    baseline_rss_bytes=180_000_000,
)


@pytest.fixture(scope="module")
def warehouse(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, int]:
    """Build the small warehouse once for the module, outside the user's cache.

    The suite caches warehouses under the user cache directory, which is right for a
    benchmark run and wrong for a test — a stale file there would make these tests
    pass against data they did not generate.
    """
    directory = tmp_path_factory.mktemp("warehouse")
    frames = generate(SMALL)
    path = directory / "warehouse.sqlite"

    from sqlalchemy import create_engine  # noqa: PLC0415

    engine = create_engine(f"sqlite:///{path}")
    rows = 0
    try:
        for source, frame in frames.items():
            frame.write_database(
                table_name=source, connection=engine, if_table_exists="replace"
            )
            rows += frame.height
    finally:
        engine.dispose()
    return path, rows


def test_rendering_distinct() -> None:
    """No two rows may look alike, or matching gets done for free.

    matchlab content-addresses records: identical field values share a leaf cluster
    before any model runs. If the generator rendered one entity the same way twice,
    those rows would arrive already merged, the matchers would have less to do than
    the case claims, and every topology would converge on the same number.
    """
    rows = pl.concat(generate(SMALL).values())

    assert rows.n_unique(subset=["name", "postcode"]) == rows.height


def test_renderings_refused() -> None:
    """Past `VARIATIONS` rows per entity, two would collide. Better to say so."""
    with pytest.raises(ValueError, match="distinct renderings"):
        Warehouse(
            n_entities=10, records_per_entity=VARIATIONS, n_sources=4, coverage=1.0
        )


def test_cleaning_collapses() -> None:
    """However a name is rendered, cleaning has to reach the same value.

    This is the other half of the pair above: distinct raw forms are only useful if
    they clean to something matchable. If cleaning stopped absorbing a suffix, the
    plans would under-merge and the correctness assertions below would start firing
    without saying why.
    """
    rows = pl.concat(generate(SMALL).values())
    cleaned = rows.select(
        pl.col("name")
        .str.to_uppercase()
        .str.replace_all(r"\s+|\.|\bLIMITED\b|\bLTD\b|\bPLC\b", ""),
        pl.col("postcode").str.to_uppercase().str.replace_all(r"\s+", ""),
        pl.col("_truth"),
    )

    assert cleaned.n_unique(subset=["name", "postcode"]) == SMALL.n_entities
    # And no two entities collapse into each other.
    assert (
        cleaned.group_by("name", "postcode")
        .agg(pl.col("_truth").n_unique().alias("entities"))["entities"]
        .max()
        == 1
    )


def test_warehouse_size() -> None:
    """`expected_rows` announces a size before anything is generated.

    Approximate on purpose — membership is a modular filter over a pseudo-random draw,
    so it lands near the coverage fraction rather than exactly on it. Within a percent
    is close enough to report, and far enough from the row counts a broken generator
    would produce to be worth asserting.
    """
    warehouse = SMALL.model_copy(update={"seed": 7})
    path, rows = ensure(warehouse)
    try:
        assert path.exists()
        assert abs(rows - warehouse.expected_rows) / warehouse.expected_rows < 0.01
    finally:
        path.unlink(missing_ok=True)
        path.with_suffix(".rows").unlink(missing_ok=True)


@pytest.mark.parametrize("topology", list(Topology))
def test_topology_resolves(topology: Topology, warehouse: tuple[Path, int]) -> None:
    """The correctness guard the timings depend on.

    `measure` asserts this itself, so this test is really checking that the assertion
    is reachable and that all four shapes build, collect and resolve. `HUB` has no
    guaranteed figure — it cannot merge two spokes that share an entity the hub never
    saw — so its shape is checked instead: strictly more entities than the shapes that
    can reach every pair.
    """
    path, rows = warehouse
    case = Case(topology=topology, warehouse=SMALL, store=Store.MEMORY)

    measurement = measure(case, path, rows)

    assert measurement.n_steps > 0
    assert measurement.collect_cold_s > 0
    if topology is Topology.HUB:
        assert measurement.n_entities_resolved > SMALL.n_entities
    else:
        assert measurement.n_entities_resolved == case.expected_entities


def test_broken_plan_fails(
    warehouse: tuple[Path, int],
) -> None:
    """Measuring a plan that stopped matching must be an error, not a fast number.

    Simulated by pointing a case at a warehouse it does not describe, which is the
    same situation from the assertion's point of view: the plan resolves a number of
    entities that the case says is impossible.
    """
    path, rows = warehouse
    mismatched = Case(
        topology=Topology.MESH,
        warehouse=SMALL.model_copy(update={"n_entities": SMALL.n_entities * 2}),
        store=Store.MEMORY,
    )

    with pytest.raises(AssertionError, match="expected"):
        measure(mismatched, path, rows)


def test_record_steps_shared(
    warehouse: tuple[Path, int],
) -> None:
    """Mesh links every pair but reads each source through one cleaned record step.

    Six links over four sources would be twelve record steps if they were not shared.
    The step count is what proves they are, and it is why the suite can claim to be
    measuring plan shape rather than accidental duplication.
    """
    path, _ = warehouse
    plan = build(Topology.MESH, declare(path, SMALL.sources))

    record_steps = [step for step in plan.lineage() if step.kind == "transform"]

    # Four cleaned record steps for the dedupes, four read back through the resolver
    # output.
    assert len(record_steps) == 2 * SMALL.n_sources


def test_collect_warm(
    warehouse: tuple[Path, int],
) -> None:
    """The warm column has to be measuring cache hits, not a second run of the work."""
    path, _ = warehouse
    store = DuckDBStore(":memory:")
    try:
        build(Topology.MESH, declare(path, SMALL.sources)).collect(
            store, interactive=False
        )
        before = store.stats()
        build(Topology.MESH, declare(path, SMALL.sources)).collect(
            store, interactive=False
        )

        assert store.stats().artifacts == before.artifacts
    finally:
        store.close()


def test_run_roundtrip(tmp_path: Path) -> None:
    """The results file exists to be diffed later, so it has to reload exactly."""
    measurement = Measurement(
        n_rows=1000,
        n_steps=24,
        collect_cold_s=1.5,
        entities_s=0.1,
        collect_warm_s=0.2,
        peak_rss_bytes=500_000_000,
        store_bytes=1_000_000,
        n_entities_resolved=300,
    )
    run = Run(
        environment=ENVIRONMENT,
        sweeps=(
            SweepResult(
                title="Plan topology",
                holding="300 entities",
                column="topology",
                results=(
                    Result(
                        case=Case(topology=Topology.MESH, warehouse=SMALL),
                        runs=(measurement,),
                    ),
                ),
            ),
        ),
    )

    path = save(run, tmp_path)

    assert Run.model_validate(json.loads(path.read_text())) == run


def _result(topology: Topology, rows: int, cold: float, entities: int) -> Result:
    """One measured case, varying only the fields the summary reads."""
    return Result(
        case=Case(topology=topology, warehouse=SMALL),
        runs=(
            Measurement(
                n_rows=rows,
                n_steps=4,
                collect_cold_s=cold,
                entities_s=0.1,
                collect_warm_s=cold / 20,
                peak_rss_bytes=180_000_000 + rows * 1000,
                store_bytes=1_000_000,
                n_entities_resolved=entities,
            ),
        ),
    )


def _summarised(*sweeps: SweepResult) -> str:
    """What `summarise` prints for these sweeps, as plain text."""
    console = Console(file=io.StringIO(), width=MIN_WIDTH, no_color=True)
    summarise(Run(environment=ENVIRONMENT, sweeps=sweeps), console)
    return console.file.getvalue()  # ty: ignore[unresolved-attribute]


TOPOLOGY_SWEEP = SweepResult(
    title="Plan topology",
    holding="300 entities",
    column="topology",
    results=(
        _result(Topology.DEDUPE, rows=1_000, cold=0.5, entities=300),
        _result(Topology.CHAIN, rows=4_000, cold=1.6, entities=300),
        _result(Topology.HUB, rows=4_000, cold=2.0, entities=320),
        _result(Topology.MESH, rows=4_000, cold=3.0, entities=300),
    ),
)


def test_summary_tabulates() -> None:
    """The topology table has to carry dedupe itself, not just ratios of it.

    Every other row is a ratio against dedupe, and a ratio whose denominator is not
    printed anywhere cannot be checked by the person reading it.
    """
    printed = _summarised(TOPOLOGY_SWEEP)

    assert "Plan topology" in printed
    assert "rows ×dedupe" in printed
    assert "dedupe" in printed

    # hub resolved 20 entities mesh did not, so it left that many unmerged
    assert "20 entities unmerged" in printed
    assert "Caching" in printed


def test_summary_omits() -> None:
    """`quick` runs one topology sweep, so the scaling section has nothing to say.

    It has to disappear rather than render an empty table: a heading over no rows
    reads as a measurement that came back blank, not one that was never taken.
    """
    printed = _summarised(TOPOLOGY_SWEEP)

    assert "Scaling" not in printed


def test_summary_scaling() -> None:
    """Which axis costs more is the comparison, so the two cannot be split apart."""
    printed = _summarised(
        TOPOLOGY_SWEEP,
        SweepResult(
            title="Number of entities",
            holding="mesh",
            column="entities",
            results=(
                _result(Topology.MESH, rows=1_000, cold=0.5, entities=300),
                _result(Topology.MESH, rows=8_000, cold=3.0, entities=2_400),
            ),
        ),
        SweepResult(
            title="Records per entity",
            holding="mesh",
            column="records",
            results=(
                _result(Topology.MESH, rows=1_000, cold=0.5, entities=300),
                _result(Topology.MESH, rows=4_000, cold=2.5, entities=300),
            ),
        ),
    )

    assert "Scaling" in printed
    assert "entities" in printed
    assert "duplicate records" in printed

    # 8x the rows for 6x the time is sublinear; 4x the rows for 5x is not
    assert "sublinear" in printed
    assert "superlinear" in printed


def test_profiles_buildable() -> None:
    """A profile that cannot be constructed fails at run time, hours in."""
    for name in ("quick", "standard", "full"):
        sweeps = profile(name, Store.FILE)
        assert sweeps
        for sweep in sweeps:
            assert sweep.cases
            for case in sweep.cases:
                assert case.warehouse.expected_rows > 0

    with pytest.raises(KeyError, match="Unknown profile"):
        profile("enormous", Store.FILE)


def test_baseline_middle() -> None:
    """Each sweep must vary one axis and leave the rest at the baseline.

    A table that held two things constant and moved two cannot answer a question, and
    the failure would be silent — the numbers would still print.
    """
    for sweep in profile("standard", Store.FILE):
        varying = {
            "Number of entities": "n_entities",
            "Records per entity": "records_per_entity",
            "Number of sources": "n_sources",
        }.get(sweep.title)

        for case in sweep.cases:
            fixed = {
                field: value
                for field, value in case.warehouse.model_dump().items()
                if field != varying
            }
            assert fixed == {
                field: value
                for field, value in BASELINE.model_dump().items()
                if field != varying
            }, f"{sweep.title} moved more than {varying}"
