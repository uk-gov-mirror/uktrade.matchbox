"""Comparison display table for entity resolution evaluation."""

from typing import Any

import polars as pl
from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import DataTable
from textual.widgets.data_table import ColumnKey

from matchlab.eval.samples import EvaluationItem
from matchlab.eval.tui.widgets.styling import get_group_style


class ComparisonDisplayTable(DataTable):
    """DataTable for comparing records with keyboard-driven assignment.

    The DataTable's own cursor column is the anchor for the visible "1"-"9" labels,
    even though the cursor itself is hidden:

    - The cursor column is labelled "1", the next column "2", and so on.
    - Paging right or left moves the cursor by 9 columns, and forces a scroll so the
      anchor column sits at the left edge, keeping the labels visible.
    """

    current_item: reactive[EvaluationItem | None] = reactive(None)
    current_assignments: reactive[dict[int, str]] = reactive({}, init=False)
    current_group: reactive[str] = reactive("")
    table_ready: reactive[bool] = reactive(False)

    # DataTable overrides
    BINDINGS = [
        Binding("up", "page_up", "Page up", show=False),
        Binding("down", "page_down", "Page down", show=False),
        Binding("right", "page_right", "Page right", show=False),
        Binding("left", "page_left", "Page left", show=False),
    ]

    def __init__(
        self, scroll_debounce_delay: float | None = 0.3, **kwargs: Any
    ) -> None:
        """Initialise the comparison display table."""
        super().__init__(**kwargs)
        self.zebra_stripes = True
        # The cursor tracks columns, for the anchor logic below, but stays hidden.
        self.cursor_type = "column"
        self.show_cursor = False
        self.fixed_columns = 1

        self._scroll_debounce_delay: float | None = scroll_debounce_delay

    def watch_current_item(self, item: EvaluationItem | None) -> None:
        """Rebuild the table when the item changes (Textual reactive pattern)."""
        self.table_ready = False

        # Reset the cursor to the start (column 1, skipping the fixed 'Field' column)
        self.move_cursor(row=0, column=1, animate=False)
        self.scroll_x = 0

        if not item:
            self.clear(columns=True)
            return

        unique_groups = item.get_unique_record_groups()

        self.clear(columns=True)
        self.add_column("Field", key="field")

        for i, _ in enumerate(unique_groups):
            # Headers will be populated by _update_headers shortly
            self.add_column("", key=f"col_{i}")

        for field in item.fields:
            row_values = [Text(field.display_name)]
            for group in unique_groups:
                record_idx = group[0]  # Representative record from group
                row_data = item.records.filter(pl.col("leaf") == record_idx).row(
                    0, named=True
                )
                # Take first non-empty value from source columns
                value = next(
                    (
                        str(row_data.get(col, "")).strip()
                        for col in field.source_columns
                        if row_data.get(col)
                    ),
                    "",
                )
                row_values.append(Text(value))
            self.add_row(*row_values, key=field.display_name)

        self.table_ready = True
        # Force header update now that columns exist
        self._update_headers()

    def watch_cursor_coordinate(self, old_value: float, new_value: float) -> None:
        """Update headers whenever the cursor moves (manual scroll or paging)."""
        self._update_headers()

    def watch_current_assignments(
        self, old_assignments: dict[int, str], new_assignments: dict[int, str]
    ) -> None:
        """Update column headers and cells when assignments change."""
        # Find which columns changed
        all_group_indices = set(old_assignments.keys()) | set(new_assignments.keys())

        for group_idx in all_group_indices:
            old_val = old_assignments.get(group_idx)
            new_val = new_assignments.get(group_idx)

            if old_val != new_val:
                self._paint_column(group_idx, new_val)

        # Refresh headers (to update status symbols)
        self._update_headers()

    def on_key(self, event: events.Key) -> None:
        """Handle keyboard shortcuts for assignment."""
        if event.key in "abcdefghijklmnopqrstuvwxyz":
            self.current_group = event.key
            self.post_message(self.CurrentGroupChanged(event.key))
            event.stop()

        elif event.key in "123456789":
            if not self.current_item or not self.current_group:
                return

            key_num = int(event.key)

            # Group index is relative to the anchor (cursor column). Key '1' is the
            # anchor itself, and the fixed field column (index 0) does not count.
            anchor_col = max(1, self.cursor_column)
            group_idx = (anchor_col - 1) + (key_num - 1)

            unique_groups = self.current_item.get_unique_record_groups()
            if 0 <= group_idx < len(unique_groups):
                self.post_message(self.AssignmentMade(group_idx, self.current_group))

            event.stop()

    def action_page_right(self) -> None:
        """Move the anchor (cursor) right by 9 columns and force scroll."""
        if not self.current_item:
            return

        unique_groups = self.current_item.get_unique_record_groups()
        total_cols = len(unique_groups) + self.fixed_columns

        current_col = self.cursor_column

        # The last page should be full (9 columns) where possible, so the anchor never
        # goes past total_cols - 9. Example: 20 columns gives max anchor 11, showing
        # columns 11-19.
        max_anchor = max(self.fixed_columns, total_cols - 9)

        target_col = min(current_col + 9, max_anchor)

        # Snapping back to max_anchor also recovers from manual scrolling that put the
        # anchor past it.
        if target_col != current_col:
            # Move the cursor without scrolling, then scroll explicitly so the anchor
            # column lands at the left edge rather than wherever Textual defaults to.
            self.move_cursor(column=target_col, animate=False, scroll=False)
            self._scroll_column_to_left(target_col)

    def action_page_left(self) -> None:
        """Move the anchor (cursor) left by 9 columns and force scroll."""
        current_col = self.cursor_column

        # Move left by 9 columns, clamped to the first data column.
        target_col = max(self.fixed_columns, current_col - 9)

        if target_col != current_col:
            self.move_cursor(column=target_col, animate=False, scroll=False)
            self._scroll_column_to_left(target_col)

    def _scroll_column_to_left(self, column_index: int) -> None:
        """Scroll so the target column is at the left edge of the viewport."""
        region = self._get_column_region(column_index)
        target_x = region.x

        # Fixed columns obscure the left of the viewport, so subtract their width to
        # land the target column just past them, not at absolute x=0.
        fixed_offset = self._get_fixed_offset()
        scroll_target = target_x - fixed_offset.left

        self.scroll_to(
            x=scroll_target,
            animate=bool(self._scroll_debounce_delay),
            duration=self._scroll_debounce_delay,
        )

    def _update_headers(self) -> None:
        """Update column headers based on current cursor position."""
        if not self.current_item or not self.table_ready:
            return

        unique_groups = self.current_item.get_unique_record_groups()

        # The anchor is the cursor column, clamped so it never indexes the fixed
        # 'Field' column.
        anchor_col = max(1, self.cursor_column)

        # Calculate the data index offset (cursor column 1 maps to data index 0)
        data_offset = anchor_col - 1

        for i, group in enumerate(unique_groups):
            # Visual position 1-9 relative to the anchor
            # i is the data index. Visual position is (i - data_offset) + 1
            visual_pos_idx = i - data_offset + 1

            visual_pos = None
            if 1 <= visual_pos_idx <= 9:
                visual_pos = visual_pos_idx

            assignment = self.current_assignments.get(i)
            new_header = self._build_header(visual_pos, group, assignment)

            # Update DataTable column label (+1 for fixed field column)
            self.ordered_columns[i + 1].label = new_header

        # DataTable caches rendered labels, so the new ones need this to show.
        self._clear_caches()
        self.refresh()

    def _paint_column(self, group_idx: int, assignment: str | None) -> None:
        """Paint or unpaint a column with colour and symbol."""
        if not self.current_item:
            return

        unique_groups = self.current_item.get_unique_record_groups()
        if group_idx >= len(unique_groups):
            return

        col_key: ColumnKey = f"col_{group_idx}"
        colour = get_group_style(assignment)[1] if assignment else None

        for row_key in self.rows:
            cell_value = str(self.get_cell(row_key, col_key))
            text_obj = Text(cell_value)
            if colour:
                text_obj.stylize(colour)
            self.update_cell(row_key, col_key, text_obj)

    def _build_header(
        self, visual_pos: int | None, group: list[int], assignment: str | None = None
    ) -> Text:
        """Build header with position, count, and optional assignment."""
        dup_count = len(group)
        count_text = f" (×{dup_count})" if dup_count > 1 else ""

        header = Text()

        if assignment:
            symbol, colour = get_group_style(assignment)
            pos_text = f" {visual_pos}" if visual_pos is not None else ""
            header.append(f"{symbol}{pos_text}{count_text}")
            header.stylize(colour)
        else:
            if visual_pos is not None:
                header.append(f"{visual_pos}{count_text}", style="dim")
            else:
                header.append(count_text.strip(), style="dim")

        return header

    class AssignmentMade(Message):
        """Posted when a single column assignment is made."""

        def __init__(self, column_idx: int, group: str) -> None:
            """Initialise assignment made message."""
            super().__init__()
            self.column_idx = column_idx
            self.group = group

    class CurrentGroupChanged(Message):
        """Posted when current group selection changes."""

        def __init__(self, group: str) -> None:
            """Initialise current group changed message."""
            super().__init__()
            self.group = group
