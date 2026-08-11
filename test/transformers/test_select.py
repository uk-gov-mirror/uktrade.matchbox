"""Tests for `Select`, projection semantics.

Ported from the pre-transformer cleaning cases that tested *dropping*: that job now
belongs to `Select`, not `Clean`.
"""

import polars as pl

from matchlab.transformers import Select


def test_select_keeps_named_drops_rest() -> None:
    """Projection keeps `id` and the named columns, and drops everything else."""
    data = pl.DataFrame(
        {"id": [1, 2, 3], "crn_company": ["A", "B", "C"], "crn_town": ["x", "y", "z"]}
    )

    result = Select("crn_company").apply(data)

    assert result.columns == ["id", "crn_company"]
    assert result["crn_company"].to_list() == ["A", "B", "C"]


def test_select_empty_yields_id_only() -> None:
    """Selecting nothing is a real projection to `id` alone, not a passthrough."""
    data = pl.DataFrame({"id": [1, 2], "crn_company": ["A", "B"]})

    assert Select().apply(data).columns == ["id"]


def test_select_ignores_id_in_columns() -> None:
    """`id` is always kept, so naming it too must not duplicate the column."""
    data = pl.DataFrame({"id": [1, 2], "crn_company": ["A", "B"]})

    assert Select("id", "crn_company").apply(data).columns == ["id", "crn_company"]
