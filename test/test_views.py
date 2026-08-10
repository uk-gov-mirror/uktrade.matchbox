"""Cleaning SQL semantics.

Ported from the pre-plan `test/client/test_queries.py::test_clean_*` block — the
function moved from `queries._clean` to `views._apply_cleaning`, and gained `group`,
but the projection contract is otherwise unchanged.
"""

import duckdb
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sqlglot.errors import ParseError

from matchlab.views import _apply_cleaning


@pytest.mark.parametrize(
    ("cleaning", "expected_columns", "expected_values"),
    [
        pytest.param(
            {"name": "foo_name"},
            ["id", "name"],
            {"name": ["A", "B", "C"]},
            id="simple_column_rename",
        ),
        pytest.param(
            {"upper_name": "upper(foo_name)"},
            ["id", "upper_name"],
            {"upper_name": ["A", "B", "C"]},
            id="simple_transformation",
        ),
        pytest.param(
            {"name": "foo_name", "is_active": "foo_status = 'active'"},
            ["id", "name", "is_active"],
            {"name": ["A", "B", "C"], "is_active": [True, False, True]},
            id="multiple_columns",
        ),
    ],
)
def test_cleaning_projects(
    cleaning: dict[str, str],
    expected_columns: list[str],
    expected_values: dict[str, list],
) -> None:
    """A cleaning renames and selects the named columns."""
    data = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "foo_name": ["A", "B", "C"],
            "foo_status": ["active", "inactive", "active"],
        }
    )

    result = _apply_cleaning(data, cleaning)

    assert result.height == 3
    assert set(result.columns) == set(expected_columns)
    for column, values in expected_values.items():
        assert result[column].to_list() == values


def test_cleaning_none_passthrough() -> None:
    """cleaning=None passes the frame through untouched."""
    data = pl.DataFrame({"id": [1, 2], "name": ["John", "Jane"], "age": [25, 30]})
    assert_frame_equal(_apply_cleaning(data, None), data)


def test_cleaning_empty_dict() -> None:
    """`{}` is a real projection selecting nothing — distinct from `None`."""
    data = pl.DataFrame(
        {"id": [1, 2, 3], "name": ["A", "B", "C"], "value": [10, 20, 30]}
    )

    result = _apply_cleaning(data, {})

    assert set(result.columns) == {"id"}
    assert result["id"].to_list() == [1, 2, 3]


def test_cleaning_drops_unreferenced() -> None:
    """Columns no cleaning names are dropped."""
    data = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["John", "Jane", "Bob"],
            "age": [25, 30, 35],
            "city": ["London", "Hull", "Stratford-upon-Avon"],
        }
    )

    result = _apply_cleaning(data, {"full_name": "name"})

    assert set(result.columns) == {"id", "full_name"}
    assert result["full_name"].to_list() == ["John", "Jane", "Bob"]


def test_cleaning_multi_column_expr() -> None:
    """A cleaning expression may combine several columns."""
    data = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "first": ["John", "Jane", "Bob"],
            "last": ["Doe", "Smith", "Johnson"],
            "salary": [50000, 60000, 55000],
        }
    )

    result = _apply_cleaning(
        data, {"name": "first || ' ' || last", "high_earner": "salary > 55000"}
    )

    assert set(result.columns) == {"id", "name", "high_earner"}
    assert result["name"].to_list() == ["John Doe", "Jane Smith", "Bob Johnson"]
    assert result["high_earner"].to_list() == [False, True, False]


def test_cleaning_complex_sql() -> None:
    """Cleaning accepts arbitrary SQL expressions."""
    data = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "price": [10.5, 20.0, 15.75],
            "quantity": [2, 1, 3],
            "category": ["A", "B", "A"],
        }
    )

    result = _apply_cleaning(
        data,
        {
            "total": "price * quantity",
            "expensive": "price > 15.0",
            "category_upper": "upper(category)",
        },
    )

    assert set(result.columns) == {"id", "total", "expensive", "category_upper"}
    assert result["total"].to_list() == [21.0, 20.0, 47.25]
    assert result["expensive"].to_list() == [False, True, True]
    assert result["category_upper"].to_list() == ["A", "B", "A"]


def test_cleaning_keeps_id_and_named() -> None:
    """`id` passes through automatically; everything else must be asked for."""
    data = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "value": [10, 20, 30],
            "status": ["active", "inactive", "pending"],
        }
    )

    result = _apply_cleaning(data, {"processed_value": "value * 2"})

    assert set(result.columns) == {"id", "processed_value"}
    assert result["processed_value"].to_list() == [20, 40, 60]


def test_group_one_row_per_id() -> None:
    """Aggregates decide how each column combines — per column, not wholesale."""
    data = pl.DataFrame(
        {
            "id": [1, 1, 2],
            "company": ["acme", "acme", "beta"],
            "town": ["london", "leeds", "hull"],
        }
    )

    result = _apply_cleaning(
        data,
        {
            "name": "any_value(company)",
            "towns": "list(distinct town)",
        },
        group=True,
    ).sort("id")

    assert result["id"].to_list() == [1, 2]
    assert result["name"].to_list() == ["acme", "beta"]
    assert sorted(result["towns"][0]) == ["leeds", "london"]
    assert result["towns"][1].to_list() == ["hull"]


def test_group_non_aggregate_raises() -> None:
    """DuckDB names the offending column, which is a better error than we'd write."""
    data = pl.DataFrame({"id": [1, 1], "town": ["london", "leeds"]})

    with pytest.raises(duckdb.BinderException, match="town"):
        _apply_cleaning(data, {"town": "town"}, group=True)


def test_cleaning_invalid_sql_raises() -> None:
    """Invalid cleaning SQL raises at build time."""
    data = pl.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
    with pytest.raises(ParseError):
        _apply_cleaning(data, {"invalid": "foo bar baz"})


def test_cleaning_across_sources() -> None:
    """Multi-source frames are already joined by the time cleaning runs."""
    data = pl.DataFrame(
        {
            "id": [1, 1, 2, 2],
            "foo_key": ["a", "a", "b", "b"],
            "bar_key": ["x", "y", "z", "w"],
            "foo_name": ["Alice", "Alice", "Bob", "Bob"],
            "bar_value": [10, 20, 30, 40],
        }
    )

    result = _apply_cleaning(data, {"combined": "foo_name || ': ' || bar_value"})

    assert set(result.columns) == {"id", "combined"}
    assert result["combined"].to_list() == [
        "Alice: 10",
        "Alice: 20",
        "Bob: 30",
        "Bob: 40",
    ]
