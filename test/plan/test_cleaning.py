"""Cleaning SQL semantics.

Ported from the pre-plan `test/client/test_queries.py::test_clean_*` block — the
function moved from `queries._clean` to `cleaning._apply_cleaning` but its contract is
unchanged, so these assertions carry over verbatim.
"""

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from sqlglot.errors import ParseError

from matchlab.cleaning import _apply_cleaning


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
def test_basic_projection(
    cleaning: dict[str, str],
    expected_columns: list[str],
    expected_values: dict[str, list],
) -> None:
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


def test_none_passes_the_frame_through() -> None:
    data = pl.DataFrame({"id": [1, 2], "name": ["John", "Jane"], "age": [25, 30]})
    assert_frame_equal(_apply_cleaning(data, None), data)


def test_empty_dict_projects_to_identifiers_only() -> None:
    """`{}` is a real projection selecting nothing — distinct from `None`."""
    data = pl.DataFrame(
        {"id": [1, 2, 3], "name": ["A", "B", "C"], "value": [10, 20, 30]}
    )

    result = _apply_cleaning(data, {})

    assert set(result.columns) == {"id"}
    assert result["id"].to_list() == [1, 2, 3]


def test_unreferenced_columns_are_dropped() -> None:
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


def test_expressions_may_reference_multiple_columns() -> None:
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


def test_complex_sql_expressions() -> None:
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


def test_leaf_id_is_passed_through_when_present() -> None:
    data = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "leaf_id": ["a", "b", "c"],
            "value": [10, 20, 30],
            "status": ["active", "inactive", "pending"],
        }
    )

    result = _apply_cleaning(data, {"processed_value": "value * 2"})

    assert set(result.columns) == {"id", "leaf_id", "processed_value"}
    assert result["leaf_id"].to_list() == ["a", "b", "c"]
    assert result["processed_value"].to_list() == [20, 40, 60]


def test_invalid_sql_raises() -> None:
    data = pl.DataFrame({"id": [1, 2, 3], "name": ["A", "B", "C"]})
    with pytest.raises(ParseError):
        _apply_cleaning(data, {"invalid": "foo bar baz"})


def test_combines_fields_across_sources() -> None:
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
