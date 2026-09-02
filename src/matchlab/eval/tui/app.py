"""Textual app that drives `review()`.

`EvaluationQueue` holds the sampled clusters still to review, always processing from
the front. `EntityResolutionApp` composes the UI, handles key bindings, and turns a
fully painted cluster into a `Judgement` on submit.
"""

import logging
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import Footer, Label

from matchlab.eval.samples import EvaluationItem, create_judgement, get_samples
from matchlab.eval.tui.modals import HelpModal, NoSamplesModal
from matchlab.eval.tui.widgets.assignment import AssignmentBar
from matchlab.eval.tui.widgets.styling import get_group_style
from matchlab.eval.tui.widgets.table import ComparisonDisplayTable
from matchlab.stores.default import default_store

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matchlab.eval.samples import ResolverRef
    from matchlab.stores import Store

logger = logging.getLogger(__name__)


class CLIEvaluationSession(BaseModel):
    """One queued cluster and the group assignments made for it so far."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    item: EvaluationItem
    assignments: dict[int, str] = Field(default_factory=dict)


class EvaluationQueue:
    """Deque-based queue with current session always at front."""

    def __init__(self) -> None:
        """Initialise the queue."""
        self.sessions: deque[CLIEvaluationSession] = deque()

    @property
    def current(self) -> CLIEvaluationSession | None:
        """Get the current session (always at index 0)."""
        return self.sessions[0] if self.sessions else None

    @property
    def total_count(self) -> int:
        """Total number of sessions in the queue."""
        return len(self.sessions)

    def skip_current(self) -> None:
        """Move the current session to the back of the queue."""
        if len(self.sessions) > 1:
            self.sessions.rotate(-1)

    def remove_current(self) -> CLIEvaluationSession | None:
        """Remove and return the current session."""
        return self.sessions.popleft() if self.sessions else None

    def add_sessions(self, sessions: list[CLIEvaluationSession]) -> int:
        """Add new sessions to the queue, skipping ones already queued.

        Returns:
            Number of unique sessions added.
        """
        if not sessions:
            return 0

        existing_ids = {tuple(sorted(session.item.leaves)) for session in self.sessions}
        unique_items = [
            session
            for session in sessions
            if tuple(sorted(session.item.leaves)) not in existing_ids
        ]

        if unique_items:
            self.sessions.extend(unique_items)

        return len(unique_items)


class EntityResolutionApp(App):
    """Main Textual application for entity resolution evaluation."""

    CSS_PATH = Path(__file__).parent / "styles.tcss"
    TITLE = "matchlab evaluate"
    SUB_TITLE = "match labelling tool"

    # Triggered by action_* methods
    BINDINGS = [
        ("shift+right", "skip", "Skip"),
        ("space", "submit", "Submit"),
        ("escape", "clear", "Clear"),
        ("question_mark,f1", "show_help", "Help"),
        ("ctrl+q,ctrl+c", "quit", "Quit"),
    ]

    # Reactive variables that trigger UI updates
    status: reactive[tuple[str, str]] = reactive(("○ Ready", "dim"))
    current_item: reactive[EvaluationItem | None] = reactive(None)
    current_assignments: reactive[dict[int, str]] = reactive({}, init=False)

    sample_limit: int
    resolver: "ResolverRef | Sequence[ResolverRef]"
    store: "Store | None"
    session_tag: str | None
    seed: int | None
    show_help: bool

    queue: EvaluationQueue
    timer: Timer | None = None

    def __init__(
        self,
        resolver: "ResolverRef | Sequence[ResolverRef]",
        num_samples: int = 5,
        store: "Store | None" = None,
        session_tag: str | None = None,
        seed: int | None = None,
        show_help: bool = False,
        scroll_debounce_delay: float | None = 0.3,
    ) -> None:
        """Initialise the entity resolution app.

        Args:
            resolver: The resolver whose clusters are reviewed, or the name one was
                published under. The second form reviews a store without the plan
                that built it. Several of either reviews their merged components.
            num_samples: Number of clusters to sample for evaluation.
            store: Where judgements are stored. Defaults to the module default.
            session_tag: String to use for tagging judgements.
            seed: Fixes which clusters the session draws, so it can be reproduced.
            show_help: Whether to show help on start.
            scroll_debounce_delay: Delay before updating column headers after scroll.
                Set to None to disable debouncing (useful for tests).
        """
        super().__init__()

        self.resolver = resolver
        self.sample_limit = num_samples
        self.store = store
        self.session_tag = session_tag
        self.seed = seed
        self._refills = 0

        self.show_help = show_help
        self._scroll_debounce_delay = scroll_debounce_delay
        self.queue = EvaluationQueue()

    # Lifecycle methods
    def compose(self) -> ComposeResult:
        """Compose the main application UI."""
        yield Vertical(
            Horizontal(
                Label("-", id="current-group-label"),
                AssignmentBar(id="assignment-bar"),
                Label(id="status-right"),
                id="status-bar",
                classes="status-bar",
            ),
            ComparisonDisplayTable(
                id="record-table", scroll_debounce_delay=self._scroll_debounce_delay
            ),
            id="main-container",
        )
        yield Footer()

    async def on_mount(self) -> None:
        """Initialise the application."""
        await self.load_samples()

        if self.queue.total_count == 0:
            await self._handle_no_samples()
            return

        self._load_current_item()

        if self.show_help:
            self.call_after_refresh(self.action_show_help)

    # Message handlers
    def on_comparison_display_table_assignment_made(
        self, message: ComparisonDisplayTable.AssignmentMade
    ) -> None:
        """Update reactive assignments when table reports an assignment."""
        # Create new dict to trigger reactivity
        new_assignments = {
            **self.current_assignments,
            message.column_idx: message.group,
        }
        self.current_assignments = new_assignments

        # Save to queue item
        if current := self.queue.current:
            current.assignments = new_assignments

        # Update assignment bar
        _, colour = get_group_style(message.group)
        assignment_bar = self.query_one("#assignment-bar", AssignmentBar)
        assignment_bar.set_position(message.column_idx, message.group.upper(), colour)

    def on_comparison_display_table_current_group_changed(
        self, message: ComparisonDisplayTable.CurrentGroupChanged
    ) -> None:
        """Update current group label when table's current group changes."""
        # Update current group label
        if message.group:
            _, colour = get_group_style(message.group)
            label_text = f"[{colour}]{message.group.upper()}[/{colour}]"
        else:
            label_text = "-"
        self.query_one("#current-group-label", Label).update(label_text)

    # Reactive watchers
    def watch_status(self) -> None:
        """React to status changes."""
        self._update_status_labels()

    def watch_current_item(self, item: EvaluationItem | None) -> None:
        """Rebuild the assignment bar for the item, and propagate it to the table."""
        table = self.query_one(ComparisonDisplayTable)
        table.current_item = item

        # Reset assignment bar for new item
        if item:
            unique_record_groups = item.get_unique_record_groups()
            num_columns = len(unique_record_groups)
            assignment_bar = self.query_one("#assignment-bar", AssignmentBar)
            assignment_bar.reset(num_columns)

            # Load existing assignments into the bar
            if self.current_assignments:
                for col_idx, group in self.current_assignments.items():
                    _, colour = get_group_style(group)
                    assignment_bar.set_position(col_idx, group.upper(), colour)

        # Reset current group label
        self.query_one("#current-group-label", Label).update("-")

    def watch_current_assignments(self, assignments: dict[int, str]) -> None:
        """Propagate assignment changes to the table."""
        table = self.query_one(ComparisonDisplayTable)
        table.current_assignments = assignments

    # Private methods
    def _load_current_item(self) -> None:
        """Load current queue item into the display."""
        current = self.queue.current
        if current:
            # Set reactive properties (will propagate to table via watchers)
            self.current_assignments = current.assignments.copy()
            self.current_item = current.item

    def _update_status_labels(self) -> None:
        """Update right status label."""
        self.query_one("#status-right", Label).update(self._build_status_right())

    def _build_status_right(self) -> str:
        """Build right status text with status indicator."""
        message, colour = self.status
        if message:
            return f"[{colour}]{message}[/]"
        return "[dim]○ Ready[/]"

    async def _handle_no_samples(self) -> None:
        """Handle empty queue state."""
        self._update_status("◯ No data", "yellow")
        logger.warning("No samples available for this resolver.")
        await self.action_show_no_samples()

    def _update_status(
        self,
        message: str,
        colour: str = "dim",
        clear_after: float | None = None,
    ) -> None:
        """Update status message with optional auto-clear.

        Args:
            message: Status message to display.
            colour: Colour for the message.
            clear_after: Seconds after which to auto-clear.
        """
        if self.timer:
            self.timer.stop()

        self.status = (message, colour)

        if clear_after:
            self.timer = self.set_timer(
                clear_after, lambda: self._update_status("○ Ready", "dim")
            )

    # Public methods
    async def load_samples(self) -> None:
        """Fetch more samples from the store until the queue reaches its limit."""
        needed = max(0, self.sample_limit - self.queue.total_count)

        if needed <= 0:
            return

        self._update_status("⚡ Fetching", "yellow")
        logger.info(
            "Fetching %s samples to reach limit of %s",
            needed,
            self.sample_limit,
        )

        # The queue is refilled as it drains, so a fixed seed would fetch the same
        # clusters every time and the leaf-based dedupe would starve it. Moving the
        # seed on per refill keeps a whole session reproducible while each fetch still
        # reaches new clusters. The dedupe drops any overlap between fetches.
        new_samples_dict = None
        try:
            new_samples_dict = get_samples(
                n=needed,
                resolver=self.resolver,
                store=self.store,
                seed=None if self.seed is None else self.seed + self._refills,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to fetch samples: {type(e).__name__}: {e}")
        finally:
            self._refills += 1

        if new_samples_dict:
            # Wrap evaluation items in CLI sessions
            sessions = [
                CLIEvaluationSession(item=item) for item in new_samples_dict.values()
            ]
            self.queue.add_sessions(sessions)

        if self.queue.total_count == 0:
            await self._handle_no_samples()
        else:
            self._update_status("✓ Ready", "green")

    # Action methods (public interface)
    async def action_skip(self) -> None:
        """Skip current entity (moves to back of queue)."""
        if self.queue.total_count > 1:
            self.queue.skip_current()
            self._load_current_item()
        else:
            await self.action_submit()
            if self.queue.total_count == 0:
                self.exit()

    async def action_submit(self) -> None:
        """Submit current entity if fully painted."""
        current = self.queue.current

        if not current:
            return

        # Check if all columns are assigned
        unique_record_groups = current.item.get_unique_record_groups()
        is_painted = len(current.assignments) == len(unique_record_groups)
        if not is_painted:
            self._update_status("⚠ Incomplete", "yellow", clear_after=3.0)
            return

        try:
            judgement = create_judgement(
                item=current.item,
                assignments=current.assignments,
                tag=self.session_tag,
            )
            store = self.store or default_store()
            store.store_judgement(judgement)
        except Exception as exc:
            self._update_status("⚠ Send failed", "red", clear_after=4.0)
            logger.exception(f"Failed to submit: {exc}")
            return

        self.queue.remove_current()
        await self.load_samples()
        self._load_current_item()
        self._update_status("✓ Sent", "green", clear_after=2.0)

    async def action_clear(self) -> None:
        """Clear current entity's group assignments."""
        if self.queue.current:
            # Clear reactive assignments (will propagate to table)
            self.current_assignments = {}

            # Update queue item
            self.queue.current.assignments = {}

            # Reset assignment bar and current group label
            if self.current_item:
                unique_record_groups = self.current_item.get_unique_record_groups()
                assignment_bar = self.query_one("#assignment-bar", AssignmentBar)
                assignment_bar.reset(len(unique_record_groups))
            self.query_one("#current-group-label", Label).update("-")

    async def action_show_help(self) -> None:
        """Show the help modal."""
        self.push_screen(HelpModal())

    async def action_show_no_samples(self) -> None:
        """Show the no samples modal."""
        self.push_screen(NoSamplesModal())

    async def action_quit(self) -> None:
        """Quit the application."""
        self.exit()
