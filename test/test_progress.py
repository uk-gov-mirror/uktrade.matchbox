"""Tests for the collection progress report.

A collection is drawn or logged, never both, so these check both shapes: that
`collect` classifies each step correctly (ran / cached / fused / failed), and that
whichever shape is in use renders what it was told.

The property that matters most is the cross-reference. Steps have no names, so a
record reading `[step 2] Ran in 0.041s` is only meaningful against a drawing that
numbers `[2]` — so exactly one of the two has to be carrying that drawing.

Fake steps throughout: a real plan's timings and step kinds are irrelevant here, and
the point is to pin the reporting independently of what is being reported.
"""

import io
import logging
import re
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import polars as pl
import pytest
from pydantic import BaseModel
from rich.console import Console
from rich.live import Live

from matchlab import lineage
from matchlab import progress as progress_module
from matchlab.adapters import (
    Adapter,
    DuckDBAdapter,
    DuckDBStoreStats,
    Fingerprint,
    StoreStats,
)
from matchlab.core import logging as mlog
from matchlab.core.kinds import StepKind
from matchlab.lineage import StepState, StepStatus, draw
from matchlab.progress import Progress, report
from matchlab.steps import Step


class _FakeSpec(BaseModel):
    """Minimal spec so the fake steps satisfy the Step contract."""

    name: str


class StoringStep(Step):
    """A step that stores a one-row artifact and counts its executions.

    Carries a `label` purely so these tests can tell nodes apart, as `FakeStep` in
    `test_lineage` does. Real steps have no such thing — they are drawn by kind and
    identified by position. It stores through `store_view`, so `VIEW` is the kind that
    matches what it actually writes.
    """

    kind: ClassVar[StepKind] = StepKind.VIEW

    def __init__(self, label: str = "fake", upstream: tuple[Step, ...] = ()) -> None:
        """Create a fake step with the given label and upstream steps."""
        self.label = label
        super().__init__(upstream=upstream)
        self.executions = 0

    def __str__(self) -> str:
        """Return the label, so drawings identify nodes by it."""
        return self.label

    @property
    def spec(self) -> BaseModel:
        """Return a minimal spec carrying only the label."""
        return _FakeSpec(name=self.label)

    def _execute(self, adapter: Adapter, fp: Fingerprint) -> None:
        self.executions += 1
        adapter.store_view(fp, pl.DataFrame({"x": [1]}))


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
    """An in-memory store for the fake steps to collect into."""
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


def _plan() -> tuple[StoringStep, StoringStep, StoringStep]:
    """A source, a view over it, and an apex. Every step stores."""
    source = StoringStep("source")
    view = StoringStep("view", upstream=(source,))
    apex = StoringStep("apex", upstream=(view,))
    return apex, source, view


def _tall_plan(depth: int) -> StoringStep:
    """A chain `depth` steps deep, which draws one row per step. Apex last."""
    step = StoringStep("step0")
    for level in range(1, depth):
        step = StoringStep(f"step{level}", upstream=(step,))
    return step


def _frame(reporter: Progress) -> str:
    """The frame as text, rendered wide and tall enough not to crop it again.

    The window has already been chosen against the patched console by the time this
    renders, so what comes back is what that terminal would have shown.
    """
    buffer = io.StringIO()
    Console(file=buffer, width=100, height=100).print(reporter._renderable())
    return buffer.getvalue()


def _run(apex: Step, store: DuckDBAdapter) -> Progress:
    """Drive a report by hand, the way `collect` does, and hand back its state."""
    steps = apex.lineage()
    with report(apex, steps, interactive=False) as reporter:
        for step in steps:
            reporter.begin(step)
            reporter.end(step, step._ensure(store))
    return reporter


def _messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records]


def _untimed(messages: list[str]) -> list[str]:
    """Messages with their durations blanked, so two runs can be compared."""
    return [re.sub(r"\d+\.\d+s", "Xs", message) for message in messages]


# -- what collect reports ------------------------------------------------------------


def test_report_first_collect_runs_all(store: DuckDBAdapter) -> None:
    """A first collect runs every storing step."""
    apex, source, view = _plan()

    state = _run(apex, store).state

    assert state[id(source)].status is StepStatus.DONE
    assert state[id(apex)].status is StepStatus.DONE
    assert state[id(view)].status is StepStatus.DONE


def test_report_recollect_cached(store: DuckDBAdapter) -> None:
    """The distinction the report exists to show: what your edit actually re-ran."""
    apex, source, _view = _plan()
    apex.collect(adapter=store, interactive=False)
    assert source.executions == 1

    state = _run(apex, store).state

    assert state[id(source)].status is StepStatus.CACHED
    assert state[id(apex)].status is StepStatus.CACHED
    assert source.executions == 1


def test_report_warm_store_all_cached(store: DuckDBAdapter) -> None:
    """Fingerprints are plan-derived, so a rebuilt plan hits the same artifacts."""
    _plan()[0].collect(adapter=store, interactive=False)

    apex, source, _view = _plan()
    state = _run(apex, store).state

    assert state[id(apex)].status is StepStatus.CACHED
    assert source.executions == 0


def test_report_failure_reraised(store: DuckDBAdapter) -> None:
    """A failing step is marked failed and the error re-raised."""
    source = StoringStep("source")
    apex = FailingStep("apex", upstream=(source,))

    with pytest.raises(RuntimeError, match="boom"):
        apex.collect(adapter=store, interactive=False)


def test_report_steps_timed(store: DuckDBAdapter) -> None:
    """Every step's duration is recorded."""
    apex, source, _view = _plan()

    reporter = _run(apex, store)

    elapsed = reporter.state[id(source)].elapsed
    assert elapsed is not None
    assert elapsed >= 0
    assert "3 steps" in reporter.summary()


def test_summary_totals_debug_counts(
    store: DuckDBAdapter,
) -> None:
    """`Cached` sits at DEBUG, so the INFO summary has to total it."""
    _plan()[0].collect(adapter=store, interactive=False)
    apex, _source, _view = _plan()

    summary = _run(apex, store).summary()

    assert "0 ran" in summary
    assert "3 cached" in summary


# -- what the summary says about the store -------------------------------------------


def _summary_of(caplog: pytest.LogCaptureFixture) -> str:
    """The last summary line `collect` logged."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("Collected")
    ][-1]


def test_summary_store_size(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Nothing else tells a user what the store costs, so the end of a run does.

    A file store, because that is the one that outlives the process and fills a disk.
    """
    store = DuckDBAdapter(tmp_path / "store.duckdb")
    apex, _source, _view = _plan()

    with caplog.at_level(logging.INFO, logger="matchlab"):
        apex.collect(adapter=store, interactive=False)
    summary = _summary_of(caplog)
    store.close()

    assert re.search(
        r"Store [\d.]+ [KMGT]?B \(\+[\d.]+ [KMGT]?B\), 3 artifacts", summary
    )


def test_summary_no_growth(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """The delta is what attributes growth to the run that caused it.

    A fully cached collect wrote nothing, and has to say so — otherwise the number is
    just the store's size again and names nothing.
    """
    store = DuckDBAdapter(tmp_path / "store.duckdb")
    _plan()[0].collect(adapter=store, interactive=False)
    apex, _source, _view = _plan()

    with caplog.at_level(logging.INFO, logger="matchlab"):
        apex.collect(adapter=store, interactive=False)
    summary = _summary_of(caplog)
    store.close()

    assert "3 cached" in summary
    assert "(+0 B)" in summary


def test_summary_omits_store_without_adapter(
    store: DuckDBAdapter,
) -> None:
    """`Progress` is constructible without one, and then simply says less."""
    apex, _source, _view = _plan()

    summary = _run(apex, store).summary()

    assert "Collected 3 steps" in summary
    assert "Store" not in summary


def test_summary_store_describes_itself(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The summary interpolates the store's own words rather than formatting them.

    What is worth saying about a store depends on the backend, so the clause comes from
    the stats object and not from the reporter. Checked by having one say something no
    real adapter would.
    """

    class Chatty(DuckDBStoreStats):
        def describe(self, since: StoreStats | None = None) -> str:
            return "a store of unusual size"

    def chatty_stats(self: DuckDBAdapter) -> Chatty:
        return Chatty(location=self.path, bytes=0)

    monkeypatch.setattr(DuckDBAdapter, "stats", chatty_stats)
    store = DuckDBAdapter(tmp_path / "store.duckdb")
    apex, _source, _view = _plan()

    with caplog.at_level(logging.INFO, logger="matchlab"):
        apex.collect(adapter=store, interactive=False)
    store.close()

    assert _summary_of(caplog).endswith(". a store of unusual size")


# -- the log channel -----------------------------------------------------------------


def test_log_plan_once_up_front(
    store: DuckDBAdapter, caplog: pytest.LogCaptureFixture
) -> None:
    """Without the tree, the numbered lines that follow refer to nothing."""
    apex, _source, _view = _plan()

    with caplog.at_level(logging.INFO, logger="matchlab"):
        apex.collect(adapter=store, interactive=False)

    headers = [m for m in _messages(caplog) if m.startswith("Collecting")]
    assert len(headers) == 1
    # One record, carrying the whole tree, the way a traceback is logged.
    assert "3 steps" in headers[0]
    assert headers[0].count("\n") == 3
    assert "└── " in headers[0]


def test_log_quotes_position(
    store: DuckDBAdapter, caplog: pytest.LogCaptureFixture
) -> None:
    """The cross-reference, end to end: every `[step N]` is an `[N]` in the drawing."""
    apex, _source, _view = _plan()

    with caplog.at_level(logging.DEBUG, logger="matchlab"):
        apex.collect(adapter=store, interactive=False)

    messages = _messages(caplog)
    tree = next(m for m in messages if m.startswith("Collecting"))
    quoted = {
        int(m) for message in messages for m in re.findall(r"\[step (\d+)\]", message)
    }

    assert quoted == set(lineage.number(apex).values())
    for position in quoted:
        assert f"[{position}] " in tree


def test_log_attributes_step_records(
    store: DuckDBAdapter, caplog: pytest.LogCaptureFixture
) -> None:
    """What a step logs while it runs is tied to the plan without it saying so.

    A `Linker` counting matches is layers below the walk that numbered its step and
    could never quote a position itself. Before this, its records were the only
    unattributed lines in a collection's log.
    """

    class ChattyStep(StoringStep):
        def _execute(self, adapter: Adapter, fp: Fingerprint) -> None:
            mlog.logger.info("Round 1: Found 13,336 matches")
            super()._execute(adapter, fp)

    apex = ChattyStep("apex", upstream=(StoringStep("source"),))

    with caplog.at_level(logging.INFO, logger="matchlab"):
        apex.collect(adapter=store, interactive=False)

    position = lineage.number(apex)[id(apex)]
    assert f"[step {position}] Round 1: Found 13,336 matches" in _messages(caplog)


def test_log_nested_restores_positions(
    store: DuckDBAdapter, caplog: pytest.LogCaptureFixture
) -> None:
    """A step can collect — `Model.results()` does — and that walk numbers its own.

    Its positions have to last exactly as long as it does. Leaking them would leave
    the outer collection quoting a tree it isn't in.
    """
    inner_apex, _source, _view = _plan()

    class NestingStep(StoringStep):
        def _execute(self, adapter: Adapter, fp: Fingerprint) -> None:
            mlog.logger.info("before")
            inner_apex.collect(adapter=adapter, interactive=False)
            mlog.logger.info("after")
            super()._execute(adapter, fp)

    apex = NestingStep("apex", upstream=(StoringStep("source"),))

    with caplog.at_level(logging.INFO, logger="matchlab"):
        apex.collect(adapter=store, interactive=False)

    outer = lineage.number(apex)[id(apex)]
    messages = _messages(caplog)
    # `after` is the one that matters: by then the inner walk has been and gone, and
    # the outer position is back in force.
    assert f"[step {outer}] before" in messages
    assert f"[step {outer}] after" in messages
    # The inner walk numbered itself against a tree of its own, logged where it began.
    assert sum("Collecting" in m for m in messages) == 2
    assert f"[step {outer}] Collecting 3 steps" in "\n".join(messages)


def test_log_prefix_outside_collection(
    store: DuckDBAdapter, caplog: pytest.LogCaptureFixture
) -> None:
    """A `Source` reads the warehouse from both, and each has its own best answer.

    Inside a walk the position is the prefix that matches the tree, so it wins. While
    a plan is still being built — `Source.index_fields` reads too — there is no
    position, and the name is what's left.
    """

    class NamedStep(StoringStep):
        def _execute(self, adapter: Adapter, fp: Fingerprint) -> None:
            mlog.logger.info("Reading from the warehouse", prefix="hmrc")
            super()._execute(adapter, fp)

    apex = NamedStep("apex", upstream=(StoringStep("source"),))

    with caplog.at_level(logging.INFO, logger="matchlab"):
        mlog.logger.info("Reading from the warehouse", prefix="hmrc")
        apex.collect(adapter=store, interactive=False)

    position = lineage.number(apex)[id(apex)]
    messages = _messages(caplog)
    assert messages[0] == "[hmrc] Reading from the warehouse"
    assert f"[step {position}] Reading from the warehouse" in messages


def test_log_prefix_released(
    store: DuckDBAdapter, caplog: pytest.LogCaptureFixture
) -> None:
    """A position outlives its walk as a lie: it numbers a tree nobody is inside."""
    apex, _source, _view = _plan()

    with caplog.at_level(logging.INFO, logger="matchlab"):
        apex.collect(adapter=store, interactive=False)
        caplog.clear()
        mlog.logger.info("afterwards")

    assert mlog.prefix.get() is None
    assert _messages(caplog) == ["afterwards"]


def test_log_prefix_released_on_failure(store: DuckDBAdapter) -> None:
    """The raise skips `end`, so releasing has to happen on the way out too."""
    apex = FailingStep("apex", upstream=(StoringStep("source"),))

    with pytest.raises(RuntimeError, match="boom"):
        apex.collect(adapter=store, interactive=False)

    assert mlog.prefix.get() is None


def test_log_level_per_outcome(
    store: DuckDBAdapter, caplog: pytest.LogCaptureFixture
) -> None:
    """Work done is INFO; structure — cached — is DEBUG."""
    _plan()[0].collect(adapter=store, interactive=False)

    # A fresh plan over the warm store with one new step on top, so a single collect
    # produces both outcomes: everything below is cached, only the new apex runs.
    warm, _source, _view = _plan()
    top = StoringStep("top", upstream=(warm,))
    positions = lineage.number(top)

    with caplog.at_level(logging.DEBUG, logger="matchlab"):
        top.collect(adapter=store, interactive=False)

    levels = {
        record.getMessage().split("]")[0].lstrip("["): record.levelno
        for record in caplog.records
        if record.getMessage().startswith("[step ")
    }
    assert levels[f"step {positions[id(warm)]}"] == logging.DEBUG  # cached
    assert levels[f"step {positions[id(top)]}"] == logging.INFO  # ran


def test_log_records_match_drawn(
    terminal: io.StringIO, caplog: pytest.LogCaptureFixture
) -> None:
    """Only the key moves between channels. What each step reports does not.

    Drawing a live tree used to demote every record to DEBUG, so a collection at a
    terminal logged nothing at all at INFO — not a step, not the summary.
    """
    # A store each, so both runs do the same work and differ only in their display.
    with caplog.at_level(logging.INFO, logger="matchlab"):
        _plan()[0].collect(adapter=DuckDBAdapter(":memory:"), interactive=True)
    drawn = _untimed(_messages(caplog))

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="matchlab"):
        _plan()[0].collect(adapter=DuckDBAdapter(":memory:"), interactive=False)
    logged = _untimed(_messages(caplog))

    assert [m for m in drawn if not m.startswith("Collecting")] == [
        m for m in logged if not m.startswith("Collecting")
    ]
    assert any("Ran in" in m for m in drawn)
    assert any(m.startswith("Collected") for m in drawn)


def test_log_not_duplicated_when_drawn(
    store: DuckDBAdapter, terminal: io.StringIO, caplog: pytest.LogCaptureFixture
) -> None:
    """The frame is already the key, and stays on screen — a second copy is noise.

    This is the interactive case. `interactive=False` is what puts the tree in the log.
    """
    apex, _source, _view = _plan()

    with caplog.at_level(logging.INFO, logger="matchlab"):
        apex.collect(adapter=store, interactive=True)

    assert not any(m.startswith("Collecting") for m in _messages(caplog))
    assert any("Ran in" in m for m in _messages(caplog))  # the records still land
    assert "apex" in terminal.getvalue()  # and the plan is on screen instead


def test_log_handler_no_smear(store: DuckDBAdapter, terminal: io.StringIO) -> None:
    """`logging.basicConfig(level=INFO)` is what people write, so it has to work.

    Its handler holds the stream it was built with, so it never sees the redirect Rich
    installs and writes into the middle of whatever is being redrawn — leaving records
    welded onto the end of a tree row.
    """
    apex, _source, _view = _plan()
    handler = logging.StreamHandler(terminal)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(message)s"))
    matchlab = logging.getLogger("matchlab")
    matchlab.addHandler(handler)
    try:
        apex.collect(adapter=store, interactive=True)
    finally:
        matchlab.removeHandler(handler)

    plain = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", terminal.getvalue())
    assert "INFO:" in plain  # the records did reach the terminal
    # Each one begins a line: `\r` is Rich returning the cursor, not a partial row.
    for offset in (m.start() for m in re.finditer("INFO:", plain)):
        assert offset == 0 or plain[offset - 1] in "\n\r"


def _on_the_frames_terminal(monkeypatch: pytest.MonkeyPatch) -> logging.StreamHandler:
    """A plain handler pointed at the same terminal the live frame draws to."""
    buffer = io.StringIO()
    monkeypatch.setattr(
        mlog, "console", Console(file=buffer, force_terminal=True, width=80, height=24)
    )
    return logging.StreamHandler(buffer)


def _writing_elsewhere(monkeypatch: pytest.MonkeyPatch) -> logging.StreamHandler:
    """A handler on some other stream, with nothing to collide with."""
    buffer = io.StringIO()
    monkeypatch.setattr(
        mlog, "console", Console(file=buffer, force_terminal=True, width=80, height=24)
    )
    return logging.StreamHandler(io.StringIO())


def _in_a_notebook(monkeypatch: pytest.MonkeyPatch) -> logging.StreamHandler:
    """A handler in a notebook, whose live display redraws in isolation."""
    buffer = io.StringIO()
    monkeypatch.setattr(mlog, "console", Console(file=buffer, force_jupyter=True))
    return logging.StreamHandler(buffer)


@pytest.mark.parametrize(
    ("configure", "borrowed"),
    [
        pytest.param(_on_the_frames_terminal, True, id="on-the-frames-terminal"),
        pytest.param(_writing_elsewhere, False, id="writing-elsewhere"),
        pytest.param(_in_a_notebook, False, id="in-a-notebook"),
    ],
)
def test_log_handler_borrowed_on_collision(
    monkeypatch: pytest.MonkeyPatch,
    configure: Callable[[pytest.MonkeyPatch], logging.StreamHandler],
    borrowed: bool,
) -> None:
    """A live frame reroutes only a handler that would write into it, then restores it.

    A plain stream handler on the same terminal collides with the redraw, so it is
    borrowed for the frame; one writing elsewhere, or a notebook's isolated display, has
    nothing to collide with and is left alone. Either way the handler belongs to the
    application and is handed back exactly as it was.
    """
    handler = configure(monkeypatch)
    original = handler.stream
    matchlab = logging.getLogger("matchlab")
    matchlab.addHandler(handler)
    try:
        with mlog.through_console():
            rerouted = handler.stream is not original
    finally:
        matchlab.removeHandler(handler)

    assert rerouted == borrowed
    assert handler.stream is original  # handed back afterwards


def test_log_console_handler_shares_terminal(
    store: DuckDBAdapter, terminal: io.StringIO
) -> None:
    """The alternative to silencing records: put them where Rich can see them.

    Printing through the console is what keeps records above the live region instead
    of through it, and it renders at full width so a logged tree stays a tree.
    """
    long_record = "Round 1: " + "x" * 60  # 69 chars, in an 80-column terminal

    class ChattyStep(StoringStep):
        def _execute(self, adapter: Adapter, fp: Fingerprint) -> None:
            mlog.logger.info(long_record)
            super()._execute(adapter, fp)

    apex = ChattyStep("apex", upstream=(StoringStep("source"),))
    handler = mlog.ConsoleHandler()
    logging.getLogger("matchlab").addHandler(handler)
    try:
        apex.collect(adapter=store, interactive=True)
    finally:
        logging.getLogger("matchlab").removeHandler(handler)

    output = terminal.getvalue()
    assert "Ran in" in output  # the log reached the same terminal as the frame
    assert "\x1b[1A" in output  # and the frame is still redrawn in place
    # Written at the console's full width, not wrapped into a narrow message column
    # the way `RichHandler` would lay it out.
    assert f"[step 1] {long_record}" in output


# -- the drawn channel ---------------------------------------------------------------


def test_tree_marks_status() -> None:
    """The drawn tree marks each step ran, cached, or failed."""
    apex, source, view = _plan()
    state = {
        id(source): StepState(StepStatus.CACHED),
        id(view): StepState(StepStatus.DONE, 0.4),
        id(apex): StepState(StepStatus.RUNNING, 2.5),
    }

    tree = draw(apex, state)

    assert "◐ [2] apex running 2.5s" in tree
    assert "● [1] view 0.4s" in tree
    assert "◍ [0] source cached" in tree


def test_tree_missing_is_pending() -> None:
    """A step absent from the state is drawn as pending."""
    apex, _source, _view = _plan()
    assert "○ [2] apex" in draw(apex, {})


def test_tree_cached_untimed() -> None:
    """`0.0s` beside a skipped step says nothing; the marker already said it."""
    assert StepState(StepStatus.CACHED, 0.0).annotation() == "cached"
    assert StepState(StepStatus.DONE, 1.24).annotation() == "1.2s"


def test_tree_escapes_markup() -> None:
    """`[7]` would otherwise parse as a style tag and blow up the render."""
    apex, _source, _view = _plan()

    markup = draw(apex, {}, markup=True)

    assert r"\[2] apex" in markup
    Console(file=io.StringIO()).print(markup)  # would raise on a bad tag


def test_tree_live_redraws_in_place(
    store: DuckDBAdapter, terminal: io.StringIO
) -> None:
    """The live display redraws in place rather than scrolling."""
    apex, _source, _view = _plan()

    apex.collect(adapter=store, interactive=True)

    output = terminal.getvalue()
    # The cursor-up sequence is what makes it one updating frame rather than a
    # tree per step.
    assert "\x1b[1A" in output
    assert "apex" in output
    assert "ran" in output  # the legend


# -- windowing a tall plan -----------------------------------------------------------


def test_window_follows_running_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """The point of a frame is to show what is happening in it."""
    _terminal(monkeypatch, height=8)
    apex = _tall_plan(12)
    steps = apex.lineage()

    with report(apex, steps, interactive=True) as reporter:
        reporter.begin(steps[0])  # the deepest node, drawn last
        deepest = _frame(reporter)
        reporter.end(steps[0], StepStatus.DONE)
        reporter.begin(apex)  # the root, drawn first
        highest = _frame(reporter)
        reporter.end(apex, StepStatus.DONE)

    assert "step0" in deepest
    assert "step11" not in deepest
    assert "step11" in highest
    assert "step0" not in highest


def test_window_shows_elision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without it the frame looks like the whole plan, and the positions look wrong."""
    _terminal(monkeypatch, height=8)
    apex = _tall_plan(12)
    steps = apex.lineage()

    with report(apex, steps, interactive=True) as reporter:
        reporter.begin(apex)
        at_root = _frame(reporter)
        reporter.end(apex, StepStatus.DONE)
        reporter.begin(steps[0])
        at_leaf = _frame(reporter)
        reporter.end(steps[0], StepStatus.DONE)

    # 12 rows, 8 of terminal, less one for the legend and one for this note.
    assert "6 more below" in at_root
    assert "above" not in at_root
    assert "6 more above" in at_leaf
    assert "below" not in at_leaf


def test_window_fits_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The note is for a frame that is hiding something; most frames aren't."""
    _terminal(monkeypatch, height=24)
    apex = _tall_plan(12)

    with report(apex, apex.lineage(), interactive=True) as reporter:
        reporter.begin(apex)
        frame = _frame(reporter)
        reporter.end(apex, StepStatus.DONE)

    assert "step0" in frame
    assert "step11" in frame
    assert "⋮" not in frame


def test_window_last_frame_whole(monkeypatch: pytest.MonkeyPatch) -> None:
    """It is what stays on screen, and what the summary line refers to.

    Nothing is being redrawn in place any more, so letting it scroll costs nothing.
    """
    output = _terminal(monkeypatch, height=8)
    apex = _tall_plan(12)

    with report(apex, apex.lineage(), interactive=True) as reporter:
        reporter.begin(apex)
        reporter.end(apex, StepStatus.DONE)
        assert "step0" not in _frame(reporter)  # windowed while it ran

    drawn = output.getvalue()
    assert "step0" in drawn
    assert "step11" in drawn


# -- choosing a channel --------------------------------------------------------------


def test_channel_tall_plan_drawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Height decides how much of the frame you see, and nothing else."""
    _terminal(monkeypatch, height=3)
    apex = _tall_plan(12)

    assert report(apex, apex.lineage(), interactive=None)._live is not None


def test_channel_window_keeps_records(
    store: DuckDBAdapter,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Only so many rows fit on screen. Every step still reports."""
    apex = _tall_plan(12)
    _terminal(monkeypatch, height=5)

    with caplog.at_level(logging.INFO, logger="matchlab"):
        apex.collect(adapter=store, interactive=True)

    ran = [m for m in _messages(caplog) if "Ran in" in m]
    assert len(ran) == 12


def test_channel_off_when_redirected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rich decides what counts as a terminal; `force_terminal=False` is its answer."""
    console = Console(file=io.StringIO(), force_terminal=False)
    monkeypatch.setattr(mlog, "console", console)
    apex, _source, _view = _plan()

    assert report(apex, apex.lineage(), interactive=None)._live is None


def test_channel_on_at_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """At a terminal, progress draws by default."""
    console = Console(file=io.StringIO(), force_terminal=True)
    monkeypatch.setattr(mlog, "console", console)
    apex, _source, _view = _plan()

    assert report(apex, apex.lineage(), interactive=None)._live is not None


def test_channel_nested_logs_not_drawn(
    terminal: io.StringIO, caplog: pytest.LogCaptureFixture
) -> None:
    """Rich allows one live display, but its positions still need a key.

    They come from a different walk, so reading them against the outer tree would
    mislead. Logging its own tree first is what makes them resolvable.
    """
    apex, _source, _view = _plan()

    with (
        caplog.at_level(logging.INFO, logger="matchlab"),
        report(apex, apex.lineage(), interactive=False),
    ):
        inner = report(apex, apex.lineage(), interactive=True)
        with inner:
            pass

    assert inner._live is None
    assert len([m for m in _messages(caplog) if m.startswith("Collecting")]) == 2
    # The flag is released by the outer report, not by the nested one.
    assert report(apex, apex.lineage(), interactive=True)._live is not None


def test_channel_failed_display_borrows_nothing(
    monkeypatch: pytest.MonkeyPatch, terminal: io.StringIO
) -> None:
    """A failed `__enter__` means no `__exit__`, so it has to clean up after itself.

    Rich raises out of `Live.start` if its first frame fails. The handlers borrowed
    just before would otherwise stay borrowed for the life of the process.
    """
    apex, _source, _view = _plan()
    handler = logging.StreamHandler(terminal)
    matchlab = logging.getLogger("matchlab")
    matchlab.addHandler(handler)
    monkeypatch.setattr(
        Live, "start", lambda *_, **__: (_ for _ in ()).throw(RuntimeError("no frame"))
    )
    try:
        with (
            pytest.raises(RuntimeError, match="no frame"),
            report(apex, apex.lineage(), interactive=True),
        ):
            pass  # never entered
    finally:
        matchlab.removeHandler(handler)

    assert handler.stream is terminal


def test_channel_unfinished_releases(
    terminal: io.StringIO,
) -> None:
    """An interrupted collection still releases its report."""
    apex, _source, _view = _plan()

    with (
        pytest.raises(RuntimeError, match="boom"),
        report(apex, apex.lineage(), interactive=True),
    ):
        raise RuntimeError("boom")

    assert report(apex, apex.lineage(), interactive=True)._live is not None
