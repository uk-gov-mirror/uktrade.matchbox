"""Progress reporting for a collection.

`collect()` walks a plan and runs the steps whose artifact is not already stored. This
module shows that walk while it runs.

**The tree is what makes a position mean anything.** Steps have no names. They are
identified by position (`matchlab.lineage.number`), so a line reading
`[step 7] Ran in 0.041s` means nothing on its own. You need the tree to know which node
`7` is. Everything here follows from that. The plan has to be put somewhere findable,
and the per-step detail has to quote the position.

**A collection draws the plan in exactly one place.** Never two.

* **A live tree**, at a terminal or in a notebook. One frame, redrawn in place, so a
  fifty-step plan doesn't scroll the shell away. The row that lights up is the row
  numbered `[7]`, so the frame carries both the key and the progress at once. It stays
  on screen throughout, and is left there in full when the run ends, so a `[step 7]` in
  the log beside it always has something to resolve against.
* **A logged tree**, when nothing is drawn, in CI, a job runner, a redirected stream.
  One multi-line record, the way a traceback is, so it can't be interleaved apart.

A drawn tree already answers what a logged one would, so a collection never prints
both. Printing the plan twice into the same screen would just be noise.

The per-step records are logged either way, each prefixed with its position, along
with a closing summary of what the collection did and what the store now costs. They
land on the `matchlab` logger.

Which of the two you get is the `interactive` argument to `Step.collect`, and that
argument is honoured as given. It is named for the assumption behind it (someone is
watching) rather than the widget it produces. Drawing means someone is watching, so the
plan can be a thing on screen that the session throws away. `interactive=False` is what
puts the tree in the log instead, for a run whose output outlives the session.

**Live output and logging share a terminal automatically.** A handler holding a raw
stream would write straight past Rich and land in the middle of the frame being
redrawn. That describes the handler `logging.basicConfig()` builds, the one nearly
every caller has. A report therefore borrows those handlers for as long as it owns the
terminal, points them at the console, and puts them back afterwards. See
`matchlab.core.logging.through_console`. Records interleave above the frame instead of
through it, with no extra setup required. `ConsoleHandler` does the same thing up
front, for a handler you are building yourself.

**A plan taller than the terminal is windowed, not cropped.** Rich would otherwise crop
a live region to its first rows (`rich.live_render`), which hides the very step you are
waiting on. The frame instead shows the rows around whichever step is running,
captioned with how much sits either side. The window follows the run down the tree,
and the last frame, with nothing left to redraw in place, shows the tree in full.

The terminal decides only how much of the frame you see. It has no say in whether a
tree is drawn, or in what the per-step records say.

`Step.collect` decides what runs, and reports what happened. This module only shows it.
"""

import logging
import time
from collections.abc import Iterable, Sequence
from contextlib import ExitStack
from contextvars import Token
from types import TracebackType
from typing import TYPE_CHECKING

from rich.console import Group, RenderableType
from rich.live import Live
from rich.text import Text

from matchlab.core import logging as mlog
from matchlab.lineage import StepState, StepStatus, draw, render
from matchlab.stores import Store, StoreStats

if TYPE_CHECKING:
    # `matchlab.steps` imports this module
    from matchlab.steps import Step

# The slice of a plan a frame shows, and a note of what it leaves out. `None` when
# the whole tree fits and nothing is left out.
_Window = tuple[list[str], Text | None]

# Whether a collection is already reporting. Rich allows one live display per console,
# and a collection can start another collection (`Model.results()` collects if needed).
# The inner one goes undrawn rather than raising. It still logs — including a tree of
# its own, immediately before the records that quote it, so its positions are read
# against its own walk and not against whatever is on screen.
_REPORTING = False

# How each outcome reads in a log line, and at what level. `Cached` is structural
# rather than work done, so it sits at `DEBUG` and the summary carries its count for
# an `INFO` reader.
_RECORDS: dict[StepStatus, tuple[int, str]] = {
    StepStatus.DONE: (logging.INFO, "Ran in {elapsed:.3f}s"),
    StepStatus.CACHED: (logging.DEBUG, "Cached"),
    StepStatus.REFRESHED: (logging.INFO, "Refreshed in {elapsed:.3f}s"),
    StepStatus.FAILED: (logging.ERROR, "Failed after {elapsed:.3f}s"),
}

# How each status reads in the legend, which has no room to spell it out.
_LABELS: dict[StepStatus, str] = {
    StepStatus.PENDING: "waiting",
    StepStatus.RUNNING: "running",
    StepStatus.DONE: "ran",
    StepStatus.CACHED: "cached",
    StepStatus.REFRESHED: "refreshed",
    StepStatus.FAILED: "failed",
}


class Progress:
    """Receives a collection's events and reports them.

    One object owns everything a running collection does besides the work itself.
    That includes what is drawn, what is logged, which position a record is
    attributed to, and, while a frame is up, where the application's own handlers
    write. They are grouped together because they must agree. A frame and a log
    disagreeing about which step is running would be worse than either on its own.

    Used as a context manager by `Step.collect`. Constructing one costs nothing and
    reports nothing. `begin` and `end` drive it.
    """

    def __init__(
        self,
        root: "Step",
        steps: Sequence["Step"],
        *,
        live: bool = False,
        nested: bool = False,
        store: Store | None = None,
    ) -> None:
        """Prepare a report for collecting `root`, whose plan is `steps`.

        Args:
            root: The step being collected, the root the tree is drawn from.
            steps: Its plan in walk order, which is what gives each step its position.
            live: Draw the tree as a live frame. Independent of what gets logged.
            nested: Whether an outer collection is already reporting. A nested report
                does not claim the console. It logs as any other does.
            store: Where the collection stores what it computes, so the summary can
                say how much room it is taking and how much this run added. Optional,
                since without one the summary simply drops that clause.
        """
        self.root = root
        self._store = store
        # The store's size before anything ran, so the summary can report what this
        # collection cost rather than only what the store now weighs. Taken in
        # `__enter__`, since a report that is built and never entered has no run to
        # attribute growth to.
        self._store_before: StoreStats | None = None
        self.steps = tuple(steps)
        # Position by step identity, the same numbering `lineage.number` gives, taken
        # from the walk `collect` already did rather than by walking again.
        self.positions = {
            id(step): position for position, step in enumerate(self.steps)
        }
        self.state: dict[int, StepState] = {
            id(step): StepState(status=StepStatus.PENDING) for step in self.steps
        }
        self._nested = nested
        self._running: Step | None = None
        self._running_since: float | None = None
        self._started: float | None = None
        # Holds the ambient prefix while a step runs, so `begin` can restore whatever
        # was in force on `end`. That matters when this report is nested, since the
        # outer collection's own step prefix was set first.
        self._prefix_token: Token[str | None] | None = None
        # The row a windowed frame centres on. Kept after a step ends rather than
        # cleared, so the view holds still between steps instead of springing back.
        self._focus: int | None = None
        # Whether the collection is over, and the frame can stop being a frame.
        self._finished = False
        # Everything this report has to undo, the frame and the log handlers it
        # borrowed to keep off it. `None` before `__enter__` and after `__exit__`.
        self._stack: ExitStack | None = None
        # Which row each step is drawn on. A plan's shape doesn't change while it
        # collects, only the markers along it, so this is fixed for the run.
        # Computed only when there is a frame to window.
        self._rows: dict[int, int] = render(root)[1] if live else {}
        # Set last, because Rich draws a first frame while constructing the display.
        self._live = (
            Live(
                get_renderable=self._renderable,
                console=mlog.console,
                refresh_per_second=8,
            )
            if live
            else None
        )

    # -- events ---------------------------------------------------------------------

    def begin(self, step: "Step") -> None:
        """Mark `step` as the one now running, and publish its position.

        Publishing the position is what attributes the records the step's own code
        emits, a linker counting matches, a source reading the warehouse. That code
        cannot name its own position, because a position belongs to the walk, not the
        step. The walk holds it here, and passes it on.
        """
        self._release()
        self._running = step
        self._running_since = time.perf_counter()
        self.state[id(step)] = StepState(status=StepStatus.RUNNING)
        self._prefix_token = mlog.prefix.set(f"step {self.positions[id(step)]}")
        self._focus = self._rows.get(id(step))
        self._refresh()

    def end(self, step: "Step", status: StepStatus) -> None:
        """Record how `step` finished, how long it took, and log a line for it."""
        elapsed = (
            time.perf_counter() - self._running_since
            if self._running_since is not None
            else 0.0
        )
        self._running = None
        self._running_since = None
        self.state[id(step)] = StepState(status=status, elapsed=elapsed)
        self._refresh()

        # Logged before the prefix is released, so this record is tagged like the ones
        # the step itself emitted.
        level, template = _RECORDS[status]
        mlog.logger.log(level, template.format(elapsed=elapsed))
        self._release()

    def _release(self) -> None:
        """Restore the prefix that was in force before the running step claimed it."""
        if self._prefix_token is not None:
            mlog.prefix.reset(self._prefix_token)
            self._prefix_token = None

    # -- lifecycle ------------------------------------------------------------------

    def __enter__(self) -> "Progress":
        """Start reporting, putting the plan on screen or in the log."""
        global _REPORTING  # one report per console
        self._started = time.perf_counter()
        if self._store is not None:
            self._store_before = self._store.stats()
        stack = ExitStack()
        try:
            if self._live is not None:
                # The redirect first, so the streams it reads are the real ones rather
                # than the proxies Rich is about to install over them. Unwinding is
                # last-in-first-out, which is also the order teardown wants. Stop the
                # frame, *then* hand the handlers back.
                stack.enter_context(mlog.through_console())
                self._live.start(refresh=True)
                stack.callback(self._stop)
            else:
                # The tree once, up front. The records that follow quote positions,
                # and this is what says which node each position is. A drawn tree is
                # already that key. It stays on screen throughout, and is left in full
                # when the run ends, so logging a second copy beside it would only
                # print the plan twice.
                mlog.logger.info(f"Collecting {len(self.steps)} steps:\n{self.tree()}")
        except BaseException:
            # Rich raises out of `start` if its first frame fails. Without this the
            # handlers it borrowed would stay borrowed for the life of the process,
            # since a failed `__enter__` means `__exit__` never runs.
            stack.close()
            raise
        self._stack = stack
        if not self._nested:
            _REPORTING = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop reporting, leaving the finished tree on screen."""
        global _REPORTING  # one report per console
        # Whatever happened, this report is no longer the one tagging records. An
        # exception out of `collect` leaves the running step's `end` uncalled.
        self._release()
        try:
            if self._stack is not None:
                self._stack.close()
                self._stack = None
        finally:
            # Even if teardown failed. Otherwise the flag would stay set, and every
            # later collection would treat itself as nested and skip drawing.
            if not self._nested:
                _REPORTING = False
        if exc_type is None:
            mlog.logger.info(self.summary())

    def _stop(self) -> None:
        """Take the frame down, leaving the finished plan on screen."""
        if self._live is None:
            return
        # Set before the stop, so the frame Rich draws on its way out is the whole tree
        # rather than the window the run was watching through.
        self._finished = True
        self._live.stop()
        # Drop the display, breaking the reference cycle it forms with this object
        # (`Live` holds `self._renderable`, a bound method, and this holds `Live`).
        # This object holds every step, so the cycle would keep a whole plan, and
        # through it the store, alive until the cyclic collector happened to run.
        # Releasing here lets the plan die with `collect`'s frame, by refcount.
        self._live = None

    def tree(self) -> str:
        """The plan, as a tree, in whatever state the collection has reached."""
        return draw(self.root, self.state)

    def summary(self) -> str:
        """One line describing what the collection did, and what it cost to keep.

        Carries the cached count because those per-step records sit at `DEBUG`. This
        is where an `INFO` reader learns what the run skipped.

        It also carries the store's size, because nothing else does. A store keeps
        everything collected into it and lives in a cache directory nobody opens, so
        the first sign of it is a full disk with nothing to blame. The delta is the
        part that assigns blame. It says what *this* run added, which is what connects
        a size to the edit that caused it.
        """
        counts = self._counts(self.state)
        elapsed = time.perf_counter() - self._started if self._started else 0.0
        # Refreshed steps are counted apart from the ones that ran because something
        # moved. They will run again next collect, so the number is what a reader needs
        # to explain a plan that never seems to settle.
        parts = [
            f"{counts[StepStatus.DONE]} ran",
            f"{counts[StepStatus.CACHED]} cached",
        ]
        if refreshed := counts[StepStatus.REFRESHED]:
            parts.append(f"{refreshed} refreshed")
        line = (
            f"Collected {len(self.steps)} steps ({', '.join(parts)}) in {elapsed:.3f}s"
        )
        if self._store is None:
            return line

        # The store says this part itself, because what is worth saying about it
        # depends on the backend. `DuckDBStoreStats` distinguishes bytes in memory
        # from bytes on a disk, and another store will have its own qualification.
        return f"{line}. {self._store.stats().describe(since=self._store_before)}"

    # -- rendering ------------------------------------------------------------------

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.refresh()

    def _renderable(self) -> Group:
        """Draw the current frame. Called by Rich on every refresh."""
        state = self._snapshot()
        # Rows come from `self._rows`, fixed at construction. State moves the markers
        # along a plan, never the shape of it.
        lines = render(self.root, state, markup=True)[0]

        # The last frame is the whole tree. It is the thing left on screen, it is what
        # the summary refers to, and letting it scroll costs nothing once there is no
        # frame left to redraw in place. Rich does the same at `Live.stop`.
        if self._finished:
            return Group(Text.from_markup("\n".join(lines)), _legend(state.values()))

        window, elided = self._window(lines)
        parts: list[RenderableType] = [Text.from_markup("\n".join(window))]
        if elided is not None:
            parts.insert(0, elided)
        parts.append(_legend(state.values()))
        return Group(*parts)

    def _window(self, lines: Sequence[str]) -> _Window:
        """The rows to show, and a note of what that leaves out.

        A plan taller than the terminal cannot be shown whole in a frame redrawn in
        place, but it doesn't have to be. What a reader needs from a running plan is
        the step that is running and the structure around it, so the frame follows it,
        keeping it centred until the ends of the tree clamp the window.

        Returns:
            `(rows, note)`, the slice to draw, and a line saying how much is hidden
            either side, or `None` when the whole tree fits and nothing is.
        """
        height = mlog.console.size.height
        if len(lines) <= height - 1:  # the legend is the only other claim on the space
            return list(lines), None

        budget = max(1, height - 2)  # the legend, and the note this returns
        focus = self._focus if self._focus is not None else 0
        # Centred on the running row, then clamped, so the window stops at the ends of
        # the tree rather than running past them into blank space.
        start = max(0, min(focus - budget // 2, len(lines) - budget))
        above, below = start, len(lines) - (start + budget)

        hidden = [
            f"{count} more {where}"
            for where, count in (("above", above), ("below", below))
            if count
        ]
        note = Text(f"⋮ {' · '.join(hidden)}", style="dim", no_wrap=True)
        return list(lines[start : start + budget]), note

    def _snapshot(self) -> dict[int, StepState]:
        """The state to draw, with the running step's elapsed time brought up to date.

        Recomputed per frame rather than stored, so the running step's timer ticks
        between events instead of freezing on whatever it read when the step started.
        """
        if self._running is None or self._running_since is None:
            return self.state
        running = time.perf_counter() - self._running_since
        return self.state | {
            id(self._running): StepState(status=StepStatus.RUNNING, elapsed=running),
        }

    @staticmethod
    def _counts(state: dict[int, StepState]) -> dict[StepStatus, int]:
        counts = {status: 0 for status in StepStatus}
        for step_state in state.values():
            counts[step_state.status] += 1
        return counts


def _legend(states: Iterable[StepState]) -> Text:
    """A key to the markers, covering only the statuses actually on screen.

    The glyphs aren't self-explanatory, and `cached` versus `ran` is the thing a reader
    most wants to know. Listing only what's present keeps it to one line and lets it
    shrink as the run settles.
    """
    present = {state.status for state in states}
    legend = Text(no_wrap=True, overflow="ellipsis")
    for status in StepStatus:
        if status not in present:
            continue
        if legend:
            legend.append("   ", style="dim")
        legend.append(status.marker, style=status.style)
        legend.append(f" {_LABELS[status]}", style="dim")
    return legend


def report(
    root: "Step",
    steps: Sequence["Step"],
    interactive: bool | None,
    store: Store | None = None,
) -> Progress:
    """Build the reporter for collecting `root`.

    Args:
        root: The step being collected.
        steps: Its plan, in walk order.
        interactive: Whether someone is watching, so the plan should be drawn rather
            than just logged. `None`, the default, takes a terminal or a notebook as
            a yes, and anything else as a no. How tall the plan is doesn't enter into
            it. One too tall to show is windowed, not cropped.
        store: Where the collection stores what it computes, so the summary can
            report the store's size and what this run added to it.
    """
    if interactive is None:
        live = mlog.console.is_terminal or mlog.console.is_jupyter
    else:
        live = interactive
    # Rich allows one live display per console, so an inner collection goes undrawn
    # whatever it asked for. It still logs, tree and all.
    live = live and not _REPORTING
    return Progress(root, steps, live=live, nested=_REPORTING, store=store)
