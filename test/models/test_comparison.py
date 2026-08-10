"""`comparison()` validates the SQL a linker compares two sides with.

A user writes the condition by hand, so its validation is the input boundary. Every
column must be qualified `l` or `r`, both sides must appear, and it must be a boolean
condition. These pin the rejections, which is what turns a typo into a clear message
rather than a confusing SQL error three steps later.
"""

import logging

import pytest
from sqlglot.errors import ParseError

from matchlab.models.comparison import comparison


def test_comparison_recompiles_a_valid_condition() -> None:
    """A well-formed condition validates and renders back to SQL for the dialect."""
    rendered = comparison("l.company = r.company", dialect="duckdb")
    assert rendered == "l.company = r.company"


def test_comparison_warns_on_or(caplog: pytest.LogCaptureFixture) -> None:
    """`OR` is valid but can't prune with an index, so it warns rather than rejects."""
    with caplog.at_level(logging.WARNING, logger="matchlab"):
        result = comparison("l.x = r.y OR l.a = r.b")

    assert result  # still returned
    assert any("Or conditions" in message for message in caplog.messages)


def test_comparison_rejects_a_subquery() -> None:
    """A comparison is a plain boolean condition, not a statement with a subquery."""
    with pytest.raises(ParseError, match="valid WHERE clause"):
        comparison("l.x = (select r.y from t)")


def test_comparison_rejects_unqualified_column() -> None:
    """A column with no `l`/`r` alias is ambiguous about which side it names."""
    with pytest.raises(ParseError, match="No table declared"):
        comparison("l.x = y")


def test_comparison_rejects_foreign_table() -> None:
    """A qualifier that isn't `l` or `r` names a side a model doesn't have."""
    with pytest.raises(ParseError, match="Found column z"):
        comparison("l.x = z.y")


def test_comparison_requires_both_sides() -> None:
    """A comparison touching only one side compares nothing across the join."""
    with pytest.raises(ParseError, match='reference both "l" and "r"'):
        comparison("l.x = l.y")
