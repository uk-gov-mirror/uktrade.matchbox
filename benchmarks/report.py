"""Say what the run found."""

import json
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console, JustifyMethod, RenderableType
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from benchmarks.measurements import Result, Run, SweepResult
from benchmarks.topologies import Topology
from matchlab.stores.base import format_bytes

# The narrowest a run may be rendered. The sweep tables carry eleven columns
# because eleven things are worth comparing side by side, and wrapping them into
# a two-line-per-row stack makes the comparison unreadable
MIN_WIDTH = 118

# One row of a summary table. Cells are plain strings unless they carry a colour.
Row = tuple[str | Text, ...]


def console() -> Console:
    """A console wide enough for the tables, whatever the terminal thinks."""
    return Console(width=max(shutil.get_terminal_size().columns, MIN_WIDTH))


def header(run: Run, console: Console) -> None:
    """Print what this run was, and on what."""
    environment = run.environment
    console.print()
    console.rule(f"[bold]matchlab benchmarks — {environment.profile}")

    facts = [
        ("matchlab", f"{environment.matchlab_version} ({environment.git_sha})"),
        ("python", f"{environment.python} on {environment.platform}"),
        ("cpus", str(environment.cpus)),
        ("store", str(environment.store)),
        ("repeats", f"{environment.repeats} (median reported)"),
        (
            "per-case floor",
            f"{environment.startup_s:.2f}s startup, "
            f"{format_bytes(environment.baseline_rss_bytes)} resident — "
            "excluded from B/row",
        ),
    ]
    width = max(len(name) for name, _ in facts)
    for name, value in facts:
        console.print(f"  [dim]{name:>{width}}[/dim]  {value}")
    console.print()


def table(sweep: SweepResult, baseline_rss: int, console: Console) -> None:
    """Print one sweep as one table."""
    rendered = Table(
        title=f"[bold]{sweep.title}[/bold]  [dim]— holding {sweep.holding}[/dim]",
        title_justify="left",
        header_style="bold",
        expand=False,
    )
    for name, justify in (
        (sweep.column, "left"),
        ("rows", "right"),
        ("steps", "right"),
        ("cold", "right"),
        ("µs/row", "right"),
        ("read", "right"),
        ("warm", "right"),
        ("peak", "right"),
        ("B/row", "right"),
        ("store", "right"),
        ("entities", "right"),
    ):
        rendered.add_column(name, justify=justify, no_wrap=True)

    for result in sweep.results:
        rendered.add_row(
            result.axis(sweep.column),
            f"{result.rows:,}",
            str(result.steps),
            f"{result.middle('collect_cold_s'):.2f}s",
            f"{result.us_per_row:.1f}",
            f"{result.middle('entities_s'):.2f}s",
            f"{result.middle('collect_warm_s'):.2f}s",
            format_bytes(int(result.middle("peak_rss_bytes"))),
            f"{result.bytes_per_row(baseline_rss):,.0f}",
            format_bytes(int(result.middle("store_bytes"))),
            f"{result.runs[0].n_entities_resolved:,}",
        )

    console.print(rendered)
    console.print()


def summarise(run: Run, console: Console) -> None:
    """State what the tables jointly show.

    Each section leads with the one line that says how to read it. A claim that repeats
    the same shape once per axis or once per topology is a table wearing prose, and is
    rendered as one; the one-off claims stay sentences, because a one-row table is
    padding. Where a profile did not measure something, the section is omitted.
    """
    console.rule("[bold]What this run found")
    console.print()

    sections = [*_scaling(run), *_topologies(run), *_caching(run)]
    if not sections:
        console.print("  [dim]Nothing to summarise: no sweep had enough cases.[/dim]\n")
        return

    for section in sections:
        console.print(section)


def _preamble(title: str, explanation: str) -> Padding:
    """A section's name, and the one line saying how to read what follows it."""
    return Padding(
        Text.from_markup(f"[bold]{title}[/bold]\n[dim]{explanation}[/dim]"),
        (0, 2, 1, 2),
    )


def _bullet(claim: Text) -> Padding:
    """One claim, as a bullet."""
    # Built up rather than started from a styled Text, whose style would be the base
    # of everything appended to it and dim the claim along with its own marker.
    bullet = Text()
    bullet.append("• ", style="dim")
    bullet.append_text(claim)
    # Padded rather than prefixed, so a claim that wraps stays inside its bullet
    # instead of falling back to the left margin and reading as a new one.
    return Padding(bullet, (0, 2, 1, 2))


def _summary_table(columns: Sequence[tuple[str, JustifyMethod]]) -> Table:
    """An untitled table on the summary's margin. `_preamble` carries its name."""
    rendered = Table(header_style="bold", expand=False)
    for name, justify in columns:
        rendered.add_column(name, justify=justify, no_wrap=True)
    return rendered


def _scaling(run: Run) -> list[RenderableType]:
    """How time and memory move as each size axis grows, both axes in one table.

    Side by side: the two axes grow the same run in different ways, and
    which of them costs more is the comparison the section exists to make.
    """
    baseline_rss = run.environment.baseline_rss_bytes
    rows = [
        *_scaling_rows(run, "Number of entities", "entities", baseline_rss),
        *_scaling_rows(run, "Records per entity", "duplicate records", baseline_rss),
    ]
    if not rows:
        return []

    rendered = _summary_table(
        (
            ("axis", "left"),
            ("measure", "left"),
            ("data", "right"),
            ("cost", "right"),
            ("shape", "left"),
            ("per row (first → last)", "right"),
        )
    )
    for row in rows:
        rendered.add_row(*row)

    return [
        _preamble(
            "Scaling",
            "What growing each axis cost, largest case against smallest. Sublinear "
            "means the per-row cost falls as the run grows. Memory is counted above "
            f"the {format_bytes(baseline_rss)} an idle process already holds.",
        ),
        Padding(rendered, (0, 2, 1, 2)),
    ]


def _scaling_rows(run: Run, title: str, axis: str, baseline_rss: int) -> list[Row]:
    """One axis' two rows: what growing it cost in time, and what it cost in memory.

    Both compare the largest case with the smallest — end to end and in that
    direction.
    """
    sweep = run.find(title)
    if sweep is None or len(sweep.results) < 2:
        return []

    first, last = sweep.results[0], sweep.results[-1]
    rows = last.rows / first.rows

    seconds = _growth(last.middle("collect_cold_s"), first.middle("collect_cold_s"))
    resident = _growth(
        last.middle("peak_rss_bytes") - baseline_rss,
        first.middle("peak_rss_bytes") - baseline_rss,
    )

    return [
        (
            axis,
            "time",
            f"{rows:.1f}x",
            f"{seconds:.1f}x",
            _shape(seconds, rows),
            f"{first.us_per_row:.1f} → {last.us_per_row:.1f} µs",
        ),
        (
            axis,
            "memory",
            f"{rows:.1f}x",
            f"{resident:.1f}x",
            _shape(resident, rows),
            f"{first.bytes_per_row(baseline_rss):,.0f} → "
            f"{last.bytes_per_row(baseline_rss):,.0f} B",
        ),
    ]


def _growth(after: float, before: float) -> float:
    """How many times over something grew. Guards the zero a fast case can produce."""
    return after / before if before > 0 else float("inf")


def _shape(observed: float, driver: float) -> Text:
    """Name a growth ratio against the growth in data that drove it.

    Sublinear is the good answer and the common one at these sizes: fixed overheads —
    opening a store, importing, priming a connection — are a smaller share of a larger
    run, so the per-row cost falls as the data grows.
    """
    ratio = observed / driver if driver else observed
    if ratio < 0.9:
        return Text("sublinear", style="green")
    if ratio <= 1.25:
        return Text("linear")
    if ratio <= 2.5:
        return Text("superlinear", style="yellow")
    return Text("steeply superlinear", style="bold red")


def _topology_row(topology: Topology, result: Result, floor: Result) -> Row:
    """One topology's cost against `dedupe`, in rows grown against time grown."""
    rows = _growth(result.rows, floor.rows)
    seconds = _growth(result.middle("collect_cold_s"), floor.middle("collect_cold_s"))
    return (
        str(topology),
        str(result.steps),
        f"{result.rows:,}",
        f"{rows:.1f}x",
        f"{seconds:.1f}x",
        _shape(seconds, rows),
        f"{result.middle('collect_cold_s'):.2f}s",
        f"{result.runs[0].n_entities_resolved:,}",
    )


def _floor_row(floor: Result) -> Row:
    """`dedupe`'s own row: the numbers every other row is a ratio of.

    Dimmed, and with no ratios of its own, because it is what the ratios are against.
    """
    return tuple(
        Text(cell, style="dim")
        for cell in (
            str(Topology.DEDUPE),
            str(floor.steps),
            f"{floor.rows:,}",
            "—",
            "—",
            "—",
            f"{floor.middle('collect_cold_s'):.2f}s",
            f"{floor.runs[0].n_entities_resolved:,}",
        )
    )


def _topologies(run: Run) -> list[RenderableType]:
    """What each plan shape costs, against the cheapest one that exists.

    `DEDUPE` reads one source; the other three read every source, so they always cover
    more rows in the same sweep. Comparing raw µs/row against that floor is not a fair
    reading — a shape landing on *more* data amortises the fixed cost of opening a
    store and connecting to the warehouse over more rows, and looks artificially
    cheaper for it, regardless of how much the shape itself adds. Comparing time
    growth against row growth — the same move `_scaling` makes across a size sweep —
    is what actually isolates the shape's cost from the volume difference.
    """
    sweep = run.find("Plan topology")
    if sweep is None:
        return []

    by_topology = {result.case.topology: result for result in sweep.results}
    floor = by_topology.get(Topology.DEDUPE)
    if floor is None:
        return []

    rendered = _summary_table(
        (
            ("topology", "left"),
            ("steps", "right"),
            ("rows", "right"),
            ("rows ×dedupe", "right"),
            ("time ×dedupe", "right"),
            ("shape", "left"),
            ("cold", "right"),
            ("entities", "right"),
        )
    )
    rendered.add_row(*_floor_row(floor))
    for topology, result in by_topology.items():
        if topology is not Topology.DEDUPE:
            rendered.add_row(*_topology_row(topology, result, floor))

    section: list[RenderableType] = [
        _preamble(
            "Plan topology",
            "Each shape against dedupe, which reads one source where the others read "
            "every one. The ratio columns hold time grown against rows grown, because "
            "a shape landing on more data spreads the fixed costs thinner and looks "
            "cheaper for it.",
        ),
        Padding(rendered, (0, 2, 1, 2)),
    ]

    hub, mesh = by_topology.get(Topology.HUB), by_topology.get(Topology.MESH)
    if hub is not None and mesh is not None:
        extra = hub.runs[0].n_entities_resolved - mesh.runs[0].n_entities_resolved
        section.append(
            _bullet(
                Text.from_markup(
                    f"hub left [bold]{extra:,} entities unmerged[/bold] that mesh "
                    "joined. A star cannot reach two spokes that share something the "
                    "hub never saw, so whatever it saves it pays for in the answer."
                )
                if extra > 0
                else Text.from_markup(
                    "hub and mesh [bold]agreed on every entity[/bold]. At this overlap "
                    "the hub can reach everything, so its cheaper shape costs "
                    "nothing — which is not true at lower coverage."
                )
            )
        )
    return section


def _caching(run: Run) -> list[RenderableType]:
    """What a second collect over an unchanged plan costs."""
    results = [result for sweep in run.sweeps for result in sweep.results]
    ratios = [
        result.middle("collect_warm_s") / result.middle("collect_cold_s")
        for result in results
        if result.middle("collect_cold_s") > 0
    ]
    if not ratios:
        return []

    # A range whose ends round to the same figure is stated once. "4.5%–4.5%" is not a
    # range
    low, high = f"{min(ratios):.1%}", f"{max(ratios):.1%}"
    span = low if low == high else f"{low}–{high}"

    return [
        _preamble(
            "Caching",
            "The price of proving nothing changed, paid on every iteration after the "
            "first.",
        ),
        _bullet(
            Text.from_markup(
                "Re-collecting an unchanged plan took "
                f"[bold]{span} of the cold run[/bold], "
                f"across all {len(ratios)} cases."
            )
        ),
    ]


def save(run: Run, directory: Path) -> Path:
    """Write the whole run to a timestamped JSON file, and return where it went.

    Every repeat is kept, not just the medians the tables showed: the file exists to be
    diffed against a later run, and a spread that has been averaged away cannot say
    whether a difference is a regression or a noisy laptop.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stamp = run.environment.started_at.replace(":", "-")
    path = directory / f"{stamp}-{run.environment.profile}.json"
    path.write_text(json.dumps(json.loads(run.model_dump_json()), indent=2) + "\n")
    return path


def now() -> str:
    """A UTC timestamp, to the second, safe in a filename."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
