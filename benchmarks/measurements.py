"""What a measurement is, and what a run of them adds up to.

Frozen and serialisable throughout, because a measurement crosses a process boundary on
its way back from the child and a run crosses a release boundary on its way into a JSON
file.
"""

from statistics import median

from pydantic import BaseModel, ConfigDict, Field

from benchmarks.cases import Case, Store


class Measurement(BaseModel):
    """What one run of one case cost.

    Attributes:
        n_rows: Rows in the warehouse it read.
        n_steps: Steps in the plan, so a shape's cost can be read per step.
        collect_cold_s: Materialising the whole plan into an empty store.
        entities_s: Reading the complete resolver output back out of it.
        collect_warm_s: Collecting the same plan again into the same store. Every step
            is a cache hit, so this is the price of establishing that nothing changed —
            the number that matters if you iterate, which matching is.
        peak_rss_bytes: The measuring process's high-water resident set, over
            everything above. Includes the store when it is in memory.
        store_bytes: What the store weighs afterwards.
        n_entities_resolved: How many entities came out. Reported for every case,
            asserted only where `Case.expected_entities` fixes it.
    """

    model_config = ConfigDict(frozen=True)

    n_rows: int
    n_steps: int
    collect_cold_s: float
    entities_s: float
    collect_warm_s: float
    peak_rss_bytes: int
    store_bytes: int
    n_entities_resolved: int


class Result(BaseModel):
    """A case and every repeat of it.

    Repeats are kept rather than averaged away: a run's spread is the only evidence
    that its middle means anything, and a JSON file that threw it out could not be
    used to argue about a regression later.
    """

    model_config = ConfigDict(frozen=True)

    case: Case
    runs: tuple[Measurement, ...] = Field(min_length=1)

    def middle(self, field: str) -> float:
        """The median of `field` across repeats.

        Median rather than mean, because the failure mode on a laptop is one run being
        interrupted by something else entirely, and a mean carries that forever.
        """
        return median(getattr(run, field) for run in self.runs)

    @property
    def rows(self) -> int:
        """Rows read. Fixed by the warehouse, so any run answers."""
        return self.runs[0].n_rows

    @property
    def steps(self) -> int:
        """Steps in the plan. Fixed by the topology, so any run answers."""
        return self.runs[0].n_steps

    @property
    def us_per_row(self) -> float:
        """Microseconds of cold collect per row read.

        The column that makes three orders of magnitude comparable: flat means linear
        scaling, and rising means something is quadratic in the data.
        """
        return self.middle("collect_cold_s") * 1_000_000 / self.rows

    def bytes_per_row(self, baseline_rss: int) -> float:
        """Resident bytes per row, above what an idle process already holds.

        The baseline has to come out, and it is not a rounding correction: a fresh
        interpreter with matchlab imported is a few hundred megabytes before it reads
        anything, which at small sizes is the entire figure. What is left is the part
        that scales with the data, and flat is what linear memory looks like.
        """
        return max(self.middle("peak_rss_bytes") - baseline_rss, 0) / self.rows

    def axis(self, column: str) -> str:
        """This case's value on the axis a sweep varies, ready to print."""
        match column:
            case "topology":
                return str(self.case.topology)
            case "entities":
                return f"{self.case.warehouse.n_entities:,}"
            case "records":
                return str(self.case.warehouse.records_per_entity)
            case "sources":
                return f"{self.case.warehouse.n_sources} / {self.case.topology}"
            case _:
                raise ValueError(f"No axis called '{column}'.")


class Environment(BaseModel):
    """The machine and the build a run happened on.

    Recorded because a benchmark result without it is not comparable with anything:
    two JSON files can only be diffed once you can see whether they came off the same
    hardware and the same commit.
    """

    model_config = ConfigDict(frozen=True)

    started_at: str
    profile: str
    repeats: int
    store: Store
    matchlab_version: str
    git_sha: str
    python: str
    platform: str
    cpus: int
    startup_s: float
    baseline_rss_bytes: int


class SweepResult(BaseModel):
    """One sweep, measured. Carries the question so the table can restate it."""

    model_config = ConfigDict(frozen=True)

    title: str
    holding: str
    column: str
    results: tuple[Result, ...] = Field(min_length=1)


class Run(BaseModel):
    """A whole run: what the machine was, and what every case cost.

    This is exactly what gets written to `results/`, so the file a later run is diffed
    against holds every raw repeat rather than the medians that were printed.
    """

    model_config = ConfigDict(frozen=True)

    environment: Environment
    sweeps: tuple[SweepResult, ...]

    def find(self, title: str) -> SweepResult | None:
        """The sweep with this title, if this run included it."""
        return next((sweep for sweep in self.sweeps if sweep.title == title), None)
