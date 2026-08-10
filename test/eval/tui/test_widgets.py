"""Review-widget display logic: the header formatting and the assignment bar.

These assert on what the widgets emit to render — header text, a dim-or-coloured
segment, the markup the bar hands Rich — rather than on the widgets' private structure,
so a behaviour-preserving refactor of either widget leaves them green.
"""

from unittest.mock import Mock

from rich.console import Console
from rich.style import Style
from rich.text import Text

from matchlab.eval.tui.widgets.assignment import AssignmentBar
from matchlab.eval.tui.widgets.table import ComparisonDisplayTable


def _rendered_styles(text: Text) -> list[Style]:
    """Styles Rich actually paints, read off rendered segments not internal spans."""
    return [segment.style for segment in Console().render(text) if segment.style]


# -- ComparisonDisplayTable header ----------------------------------------------------


def test_header_formats() -> None:
    """Position, a duplicate count once there's more than one, and the group symbol."""
    table = ComparisonDisplayTable()

    assert table._build_header(1, [1], None).plain == "1"
    assert table._build_header(2, [1, 2, 3], None).plain == "2 (×3)"

    assigned = table._build_header(2, [1, 2], "b")
    assert "2 (×2)" in assigned.plain
    # A group symbol leads the header; which glyph is `get_group_style`'s business.
    assert assigned.plain.startswith(("◇", "⬤"))


def test_header_dim_vs_coloured() -> None:
    """The visible cue a column is judged: dim until it carries a group colour."""
    table = ComparisonDisplayTable()

    unassigned = _rendered_styles(table._build_header(1, [1], None))
    assert any(style.dim for style in unassigned)

    assigned = _rendered_styles(table._build_header(1, [1], "a"))
    assert not any(style.dim for style in assigned)
    assert any(style.color for style in assigned)


# -- AssignmentBar --------------------------------------------------------------------


def test_bar_starts_empty() -> None:
    """A fresh bar holds no positions."""
    assert AssignmentBar().positions == []


def test_bar_reset() -> None:
    """Reset lays out that many empty slots."""
    bar = AssignmentBar()
    bar.update = Mock()  # Bar.update renders; these tests only read the state it holds.

    bar.reset(5)

    assert bar.positions == [None] * 5


def test_bar_set_position() -> None:
    """Setting a position records its group letter and colour."""
    bar = AssignmentBar()
    bar.update = Mock()
    bar.reset(3)

    bar.set_position(1, "a", "red")

    assert bar.positions[0] is None
    assert bar.positions[1].letter == "a"
    assert bar.positions[1].colour == "red"


def test_bar_renders_letters_and_dots() -> None:
    """A group's first slot shows its letter; the next in that group is a dot."""
    bar = AssignmentBar()
    bar.update = Mock()

    bar.reset(3)
    bar.update.assert_called_with("[dim]•[/dim][dim]•[/dim][dim]•[/dim]")

    bar.set_position(1, "a", "red")
    bar.update.assert_called_with("[dim]•[/dim][red]a[/red][dim]•[/dim]")

    # Adjacent same group: the letter is not repeated, a dot stands in.
    bar.set_position(2, "a", "red")
    bar.update.assert_called_with("[dim]•[/dim][red]a[/red][red]•[/red]")

    # A new group next to it shows its own letter.
    bar.set_position(0, "b", "blue")
    bar.update.assert_called_with("[blue]b[/blue][red]a[/red][red]•[/red]")
