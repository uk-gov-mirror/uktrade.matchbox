"""Tests for the collection progress report.

The live tree and the log channel are the same information in two shapes, so these
check both: that `collect` classifies each step correctly (ran / cached / fused /
failed), and that each channel renders what it was told.

The property that matters most is the cross-reference. Steps have no names, so a
record reading `[step 2] Ran in 0.041s` is only meaningful against a drawing that
numbers `[2]` — whichever channel is in use has to put that drawing somewhere.

Fake steps throughout: a real plan's timings and step kinds are irrelevant here, and
the point is to pin the reporting independently of what is being reported.
"""

from __future__ import annotations

import io
import logging
import re
from typing import ClassVar

import polars as pl
import pytest
from pydantic import BaseModel
from rich.console import Console

from matchlab import lineage
from matchlab import progress as progress_module
from matchlab.adapters import Adapter, DuckDBAdapter, Fingerprint
from matchlab.core import logging as mlog
from matchlab.lineage import StepState, StepStatus, draw
from matchlab.progress import Progress, report
from matchlab.steps import Step


class _FakeConfig(BaseModel):
    """Minimal config so the fake steps satisfy the Step contract."""

    name: str


class StoringStep(Step):
    """A step that stores a one-row artifact and counts its executions.

    Carries a `label` purely so these tests can tell nodes apart, as `FakeStep` in
    `test_lineage` does. Real steps have no such thing — they are drawn by kind and
    identified by position.
    """

    kind: ClassVar[str] = "fake"

    def __init__(self, label: str = "fake", upstream: tuple[Step, ...] = ()) -> None:
        self.label = label
        super().__init__(upstream=upstream)
        self.executions = 0

    def __str__(self) -> str:
        return self.label

    @property
    def config(self) -> BaseModel:
        return _FakeConfig(name=self.label)

    def _execute(self, adapter: Adapter, fp: Fingerprint) -> None:
        self.executions += 1
        adapter.store_clean(fp, pl.DataFrame({"x": [1]}))


class FusedStep(StoringStep):
    """A step that stores nothing, like a `View` feeding a model."""

    stores: ClassVar[bool] = False

    def _execute(self, adapter: Adapter, fp: Fingerprint) -> None:  # pragma: no cover
        raise AssertionError("a fused step never executes")


class FailingStep(StoringStep):
    """A step that always raises."""

    def _execute(self, adapter: Adapter, fp: Fingerprint) -> None:
        raise RuntimeError("boom")


@pytest.fixture(autouse=True)
def _no_report_leaks() -> None:
    """Clear the module flag, so one test's failure can't mute the next one."""
    progress_module._REPORTING = False


@pytest.fixture
def store() -> DuckDBAdapter:
    return DuckDBAdapter(":memory:")


def _terminal(monkeypatch: pytest.MonkeyPatch, height: int) -> io.StringIO:
    """Point matchlab at a console that renders to a buffer, and hand back the buffer.

    The buffer rather than the console, because what these tests read is what got
    drawn — and `Console.file` is typed as a plain `IO[str]`.
    """
    buffer = io.StringIO()
    console = Console(file=buffer, force_terminal=True, width=80, height=height)
    monkeypatch.setattr(mlog, "console", console)
    return buffer


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """A terminal tall enough for these plans, rendering to an inspectable buffer."""
    return _terminal(monkeypatch, height=24)


def _plan() -> tuple[StoringStep, StoringStep, FusedStep]:
    """A source, a fused view over it, and an apex that stores."""
    source = StoringStep("source")
    view = FusedStep("view", upstream=(source,))
    apex = StoringStep("apex", upstream=(view,))
    return apex, source, view


def _run(apex: Step, store: DuckDBAdapter) -> Progress:
    """Drive a report by hand, the way `collect` does, and hand back its state."""
    steps = apex.lineage()
    with report(apex, steps, progress=False) as reporter:
        for step in steps:
            reporter.begin(step)
            reporter.end(step, step._ensure(store))
    return reporter


def _messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records]


# -- what collect reports ------------------------------------------------------------


def test_a_first_collect_runs_every_storing_step(store: DuckDBAdapter) -> None:
    apex, source, view = _plan()

    state = _run(apex, store).state

    assert state[id(source)].status is StepStatus.DONE
    assert state[id(apex)].status is StepStatus.DONE
    assert state[id(view)].status is StepStatus.FUSED


def test_recollecting_reports_cached_not_ran(store: DuckDBAdapter) -> None:
    """The distinction the report exists to show: what your edit actually re-ran."""
    apex, source, _view = _plan()
    apex.collect(adapter=store, progress=False)
    assert source.executions == 1

    state = _run(apex, store).state

    assert state[id(source)].status is StepStatus.CACHED
    assert state[id(apex)].status is StepStatus.CACHED
    assert source.executions == 1


def test_a_fresh_plan_over_a_warm_store_is_all_cached(store: DuckDBAdapter) -> None:
    """Fingerprints are plan-derived, so a rebuilt plan hits the same artifacts."""
    _plan()[0].collect(adapter=store, progress=False)

    apex, source, _view = _plan()
    state = _run(apex, store).state

    assert state[id(apex)].status is StepStatus.CACHED
    assert source.executions == 0


def test_a_failure_is_marked_and_reraised(store: DuckDBAdapter) -> None:
    source = StoringStep("source")
    apex = FailingStep("apex", upstream=(source,))

    with pytest.raises(RuntimeError, match="boom"):
        apex.collect(adapter=store, progress=False)


def test_every_step_is_timed(store: DuckDBAdapter) -> None:
    apex, source, _view = _plan()

    reporter = _run(apex, store)

    elapsed = reporter.state[id(source)].elapsed
    assert elapsed is not None
    assert elapsed >= 0
    assert "3 steps" in reporter.summary()


def test_the_summary_carries_the_counts_a_debug_reader_would_have_seen(
    store: DuckDBAdapter,
) -> None:
    """`Cached` and `Fused` sit at DEBUG, so the INFO summary has to total them."""
    _plan()[0].collect(adapter=store, progress=False)
    apex, _source, _view = _plan()

    summary = _run(apex, store).summary()

    assert "0 ran" in summary
    assert "2 cached" in summary
    assert "1 fused" in summary


# -- the log channel -----------------------------------------------------------------


def test_the_plan_is_logged_once_up_front(
    store: DuckDBAdapter, caplog: pytest.LogCaptureFixture
) -> None:
    """Without the tree, the numbered lines that follow refer to nothing."""
    apex, _source, _view = _plan()

    with caplog.at_level(logging.INFO, logger="matchlab"):
        apex.collect(adapter=store, progress=False)

    headers = [m for m in _messages(caplog) if m.startswith("Collecting")]
    assert len(headers) == 1
    # One record, carrying the whole tree, the way a traceback is logged.
    assert "3 steps" in headers[0]
    assert headers[0].count("\n") == 3
    assert "└── " in headers[0]


def test_a_log_line_quotes_a_position_the_logged_tree_numbers(
    store: DuckDBAdapter, caplog: pytest.LogCaptureFixture
) -> None:
    """The cross-reference, end to end: every `[step N]` is an `[N]` in the drawing."""
    apex, _source, _view = _plan()

    with caplog.at_level(logging.DEBUG, logger="matchlab"):
        apex.collect(adapter=store, progress=False)

    messages = _messages(caplog)
    tree = next(m for m in messages if m.startswith("Collecting"))
    quoted = {
        int(m) for message in messages for m in re.findall(r"\[step (\d+)\]", message)
    }

    assert quoted == set(lineage.number(apex).values())
    for position in quoted:
        assert f"[{position}] " in tree


def test_each_outcome_logs_at_the_level_it_deserves(
    store: DuckDBAdapter, caplog: pytest.LogCaptureFixture
) -> None:
    """Work done is INFO; structure — cached, fused — is DEBUG."""
    apex, _source, view = _plan()
    positions = lineage.number(apex)

    with caplog.at_level(logging.DEBUG, logger="matchlab"):
        apex.collect(adapter=store, progress=False)

    levels = {
        record.getMessage().split("]")[0].lstrip("["): record.levelno
        for record in caplog.records
        if record.getMessage().startswith("[step ")
    }
    assert levels[f"step {positions[id(view)]}"] == logging.DEBUG  # fused
    assert levels[f"step {positions[id(apex)]}"] == logging.INFO  # ran


def test_the_live_display_keeps_its_logs_out_of_the_way(
    store: DuckDBAdapter, terminal: io.StringIO, caplog: pytest.LogCaptureFixture
) -> None:
    """A handler on the same terminal would smear the frame, so these drop to DEBUG."""
    apex, _source, _view = _plan()

    with caplog.at_level(logging.DEBUG, logger="matchlab"):
        apex.collect(adapter=store, progress=True)

    levels = {
        record.levelno for record in caplog.records if "Ran in" in record.getMessage()
    }
    assert levels == {logging.DEBUG}


# -- the drawn channel ---------------------------------------------------------------


def test_the_tree_marks_each_status() -> None:
    apex, source, view = _plan()
    state = {
        id(source): StepState(StepStatus.CACHED),
        id(view): StepState(StepStatus.FUSED),
        id(apex): StepState(StepStatus.RUNNING, 2.5),
    }

    tree = draw(apex, state)

    assert "◐ [2] apex running 2.5s" in tree
    assert "◌ [1] view fused" in tree
    assert "◍ [0] source cached" in tree


def test_a_step_missing_from_the_state_is_drawn_as_pending() -> None:
    apex, _source, _view = _plan()
    assert "○ [2] apex" in draw(apex, {})


def test_a_cached_step_is_not_given_a_time() -> None:
    """`0.0s` beside a skipped step says nothing; the marker already said it."""
    assert StepState(StepStatus.CACHED, 0.0).annotation() == "cached"
    assert StepState(StepStatus.DONE, 1.24).annotation() == "1.2s"


def test_drawing_with_markup_escapes_the_position() -> None:
    """`[7]` would otherwise parse as a style tag and blow up the render."""
    apex, _source, _view = _plan()

    markup = draw(apex, {}, markup=True)

    assert r"\[2] apex" in markup
    Console(file=io.StringIO()).print(markup)  # would raise on a bad tag


def test_the_live_display_redraws_in_place(
    store: DuckDBAdapter, terminal: io.StringIO
) -> None:
    apex, _source, _view = _plan()

    apex.collect(adapter=store, progress=True)

    output = terminal.getvalue()
    # The cursor-up sequence is what makes it one updating frame rather than a
    # tree per step.
    assert "\x1b[1A" in output
    assert "apex" in output
    assert "ran" in output  # the legend


# -- choosing a channel --------------------------------------------------------------


def test_a_plan_taller_than_the_terminal_uses_the_log_channel(
    store: DuckDBAdapter,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Cropping the tree would lose the positions the log lines quote."""
    output = _terminal(monkeypatch, height=3)
    apex, _source, _view = _plan()

    with caplog.at_level(logging.INFO, logger="matchlab"):
        apex.collect(adapter=store, progress=True)

    assert output.getvalue() == ""  # nothing was drawn
    assert any(m.startswith("Collecting") for m in _messages(caplog))


def test_progress_defaults_to_off_when_output_is_redirected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rich decides what counts as a terminal; `force_terminal=False` is its answer."""
    console = Console(file=io.StringIO(), force_terminal=False)
    monkeypatch.setattr(mlog, "console", console)
    apex, _source, _view = _plan()

    assert report(apex, apex.lineage(), progress=None)._live is None


def test_progress_defaults_to_on_at_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    console = Console(file=io.StringIO(), force_terminal=True)
    monkeypatch.setattr(mlog, "console", console)
    apex, _source, _view = _plan()

    assert report(apex, apex.lineage(), progress=None)._live is not None


def test_a_nested_collection_prints_no_tree_of_its_own(
    terminal: io.StringIO, caplog: pytest.LogCaptureFixture
) -> None:
    """Its positions come from another walk, so numbering them against the tree lies.

    The outer report uses the log channel here, so it does print a tree — which is
    what makes a second one detectable.
    """
    apex, _source, _view = _plan()

    with (
        caplog.at_level(logging.INFO, logger="matchlab"),
        report(apex, apex.lineage(), progress=False),
    ):
        inner = report(apex, apex.lineage(), progress=True)
        with inner:
            pass

    assert inner._live is None
    assert len([m for m in _messages(caplog) if m.startswith("Collecting")]) == 1
    # The flag is released by the outer report, not by the nested one.
    assert report(apex, apex.lineage(), progress=True)._live is not None


def test_an_unfinished_collection_releases_the_report(
    terminal: io.StringIO,
) -> None:
    apex, _source, _view = _plan()

    with (
        pytest.raises(RuntimeError, match="boom"),
        report(apex, apex.lineage(), progress=True),
    ):
        raise RuntimeError("boom")

    assert report(apex, apex.lineage(), progress=True)._live is not None
