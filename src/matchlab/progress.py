"""Progress reporting for a collection.

`collect()` walks a plan and runs the steps whose artifact isn't already stored. This
module is how that walk is *shown*.

**The drawing is the key to the output.** Steps have no names; they are identified by
position (`matchlab.lineage.number`), so a line reading `[step 7] Ran in 0.041s` means
nothing on its own — you need the tree to know which node `7` is. Everything here
follows from that: whichever channel is in use has to put the plan on screen, and the
per-step detail has to quote the position.

Two channels, because they suit different places:

* **The live tree** is for a terminal. One frame, redrawn in place, so a fifty-step
  plan doesn't scroll the shell away: the tree *is* the key and the progress at once,
  because the row that lights up is the row numbered `[7]`.
* **The log channel** is for everywhere else — CI, a job runner, a file. It emits the
  same tree **once**, up front, and then one record per step beneath it, each prefixed
  with its position. The tree is a single multi-line record, the way a traceback is,
  so it can't be interleaved apart. Records land on the `matchlab` logger and are
  silent unless you have configured logging.

The log channel also takes over when a plan is taller than the terminal, since a tree
that cannot be redrawn in place cannot be a live frame — and cropping it would lose
the very positions the log lines quote.

Nothing here decides *what* runs; `Step.collect` does that and reports what happened.
"""

from __future__ import annotations

import logging
import time
from types import TracebackType
from typing import TYPE_CHECKING

from rich.console import Group
from rich.live import Live
from rich.text import Text

from matchlab.core import logging as mlog
from matchlab.lineage import StepState, StepStatus, draw

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from matchlab.steps import Step

# Whether a collection is already reporting. Rich allows one live display per console,
# and a collection can start another collection (`Model.results()` collects if needed).
# The inner one stays quiet rather than raising: its positions come from a different
# walk, so numbering them against the tree already on screen would be a lie.
_REPORTING = False

#: How each outcome reads in a log line, and at what level. `Cached` and `Fused` are
#: structural rather than work done, so they sit at `DEBUG` and the summary carries
#: their counts for an `INFO` reader.
_RECORDS: dict[StepStatus, tuple[int, str]] = {
    StepStatus.DONE: (logging.INFO, "Ran in {elapsed:.3f}s"),
    StepStatus.CACHED: (logging.DEBUG, "Cached"),
    StepStatus.FUSED: (logging.DEBUG, "Fused into its consumer"),
    StepStatus.FAILED: (logging.ERROR, "Failed after {elapsed:.3f}s"),
}

#: How each status reads in the legend, which has no room to spell it out.
_LABELS: dict[StepStatus, str] = {
    StepStatus.PENDING: "waiting",
    StepStatus.RUNNING: "running",
    StepStatus.DONE: "ran",
    StepStatus.CACHED: "cached",
    StepStatus.FUSED: "fused",
    StepStatus.FAILED: "failed",
}


class Progress:
    """Receives a collection's events and renders them.

    Used as a context manager by `Step.collect`. Constructing one costs nothing and
    reports nothing; `begin` and `end` drive it.
    """

    def __init__(
        self,
        root: Step,
        steps: Sequence[Step],
        *,
        live: bool = False,
        nested: bool = False,
    ) -> None:
        """Prepare a report for collecting `root`, whose plan is `steps`.

        Args:
            root: The step being collected — the root the tree is drawn from.
            steps: Its plan in walk order, which is what gives each step its position.
            live: Draw the tree as a live frame rather than using the log channel.
            nested: Whether an outer collection is already reporting. A nested report
                neither claims the console nor prints a tree of its own.
        """
        self.root = root
        self.steps = tuple(steps)
        #: Position by step identity — the same numbering `lineage.number` gives, taken
        #: from the walk `collect` already did rather than by walking again.
        self.positions = {
            id(step): position for position, step in enumerate(self.steps)
        }
        self.state: dict[int, StepState] = {
            id(step): StepState(StepStatus.PENDING) for step in self.steps
        }
        self._nested = nested
        # Records drop to DEBUG when a live frame owns the terminal, since a handler
        # writing to it would smear the frame being redrawn; and when nested, since
        # positions from a different walk would not match the tree on screen.
        self._quiet = nested or live
        self._running: Step | None = None
        self._running_since: float | None = None
        self._started: float | None = None
        # Set last: Rich draws a first frame while constructing the display.
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

    def begin(self, step: Step) -> None:
        """Mark `step` as the one now running."""
        self._running = step
        self._running_since = time.perf_counter()
        self.state[id(step)] = StepState(StepStatus.RUNNING)
        self._refresh()

    def end(self, step: Step, status: StepStatus) -> None:
        """Record how `step` finished, how long it took, and log a line for it."""
        elapsed = (
            time.perf_counter() - self._running_since
            if self._running_since is not None
            else 0.0
        )
        self._running = None
        self._running_since = None
        self.state[id(step)] = StepState(status, elapsed)
        self._refresh()

        level, template = _RECORDS[status]
        self._log(template.format(elapsed=elapsed), level, step=step)

    # -- lifecycle ------------------------------------------------------------------

    def __enter__(self) -> Progress:
        """Start reporting, putting the plan on screen."""
        global _REPORTING  # noqa: PLW0603 - one report per console
        self._started = time.perf_counter()
        if self._live is not None:
            self._live.start(refresh=True)
        elif not self._nested:
            # The tree once, up front: the log lines that follow quote positions, and
            # this is the only thing that says which node each position is.
            self._log(f"Collecting {len(self.steps)} steps:\n{self.tree()}")
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
        global _REPORTING  # noqa: PLW0603 - one report per console
        if self._live is not None:
            self._live.stop()
            # Drop the display, breaking the reference cycle it forms with this object
            # (`Live` holds `self._renderable`, a bound method, and this holds `Live`).
            # This object holds every step, so the cycle would keep a whole plan — and
            # through it the adapter — alive until the cyclic collector happened to run.
            # Releasing here lets the plan die with `collect`'s frame, by refcount.
            self._live = None
        if not self._nested:
            _REPORTING = False
        if exc_type is None:
            self._log(self.summary())

    def tree(self) -> str:
        """The plan, as a tree, in whatever state the collection has reached."""
        return draw(self.root, self.state)

    def summary(self) -> str:
        """One line describing what the collection did.

        Carries the cached and fused counts because their per-step records sit at
        `DEBUG` — this is where an `INFO` reader learns what the run skipped.
        """
        counts = self._counts(self.state)
        elapsed = time.perf_counter() - self._started if self._started else 0.0
        return (
            f"Collected {len(self.steps)} steps ("
            f"{counts[StepStatus.DONE]} ran, "
            f"{counts[StepStatus.CACHED]} cached, "
            f"{counts[StepStatus.FUSED]} fused) in {elapsed:.3f}s"
        )

    # -- rendering ------------------------------------------------------------------

    def _log(
        self,
        message: str,
        level: int = logging.INFO,
        *,
        step: Step | None = None,
    ) -> None:
        """Log `message`, quoting `step`'s position so it is findable in the tree."""
        if self._quiet:
            level = logging.DEBUG
        prefix = f"step {self.positions[id(step)]}" if step is not None else None
        mlog.logger.log(level, message, prefix=prefix)

    def _refresh(self) -> None:
        if self._live is not None:
            self._live.refresh()

    def _renderable(self) -> Group:
        """Draw the current frame. Called by Rich on every refresh."""
        state = self._snapshot()
        tree = draw(self.root, state, markup=True)
        return Group(Text.from_markup(tree), _legend(state.values()))

    def _snapshot(self) -> dict[int, StepState]:
        """The state to draw, with the running step's elapsed time brought up to date.

        Recomputed per frame rather than stored, so the running step's timer ticks
        between events instead of freezing on whatever it read when the step started.
        """
        if self._running is None or self._running_since is None:
            return self.state
        running = time.perf_counter() - self._running_since
        return self.state | {
            id(self._running): StepState(StepStatus.RUNNING, running),
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


def _fits(root: Step) -> bool:
    """Whether `root`'s tree, plus its legend, fits the console.

    Checked once up front rather than per frame: a plan's shape doesn't change while it
    collects, so the answer can't either.
    """
    return draw(root).count("\n") + 2 <= mlog.console.size.height


def report(root: Step, steps: Sequence[Step], progress: bool | None) -> Progress:
    """Build the reporter for collecting `root`.

    Args:
        root: The step being collected.
        steps: Its plan, in walk order.
        progress: Whether to draw the live tree. `None` — the default — draws it when
            the console is a terminal or a notebook, and not otherwise. A plan too tall
            for the console uses the log channel however this is set, since a tree that
            can't be redrawn in place can't be a live frame.
    """
    if progress is None:
        live = mlog.console.is_terminal or mlog.console.is_jupyter
    else:
        live = progress
    live = live and not _REPORTING and _fits(root)
    return Progress(root, steps, live=live, nested=_REPORTING)
