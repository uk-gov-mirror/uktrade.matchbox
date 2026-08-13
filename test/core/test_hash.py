"""Content-addressing: the hash a fingerprint is built from.

`hash_dataframe` promises order-invariance. A table hashes by what it contains, not
the order rows, columns, or list elements happen to arrive in. It still changes the
hash for any real change to content. `as_sorted_list` extends that to a set of
columns, so `(1, 2)` and `(2, 1)` count as the same pair. Both hash methods must honour
the same contract, so every test runs over both.
"""

import polars as pl
import pytest

from matchlab.core.hash import HashMethod, hash_dataframe, hash_rows

methods = pytest.mark.parametrize(
    "method",
    [
        pytest.param(HashMethod.SHA256, id="sha256"),
        pytest.param(HashMethod.XXH3_128, id="xxh3_128"),
    ],
)


# -- row hashing ----------------------------------------------------------------------


@methods
def test_hash_rows_all_dtypes(method: HashMethod) -> None:
    """One hash per row over the full spread of dtypes a source can present.

    The dtype assertions guard the fixture. They show this really exercises the
    binary, struct, and list branches of `_process_column_for_hashing`, not three string
    columns in disguise.
    """
    data = pl.DataFrame(
        {
            "string_col": ["abc", "def", "ghi"],
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "struct_col": [{"a": 1, "b": "x"}, {"a": 2, "b": None}, {"a": 3, "b": "z"}],
            "binary_col": [b"data1", b"data2", b"data3"],
            "list_col": [["tag1", "tag2"], ["tag3"], ["tag4", "tag5"]],
        }
    )

    assert isinstance(data["struct_col"].dtype, pl.Struct)
    assert isinstance(data["binary_col"].dtype, pl.Binary)
    assert isinstance(data["list_col"].dtype, pl.List)

    hashes = hash_rows(data, columns=data.columns, method=method)

    assert hashes.len() == data.height


# -- order invariance -----------------------------------------------------------------


@methods
def test_hash_ignores_row_and_field_order(method: HashMethod) -> None:
    """The same content in any column or row order is the same table to the hash."""
    original = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    field_reordered = pl.DataFrame({"b": [4, 5, 6], "a": [1, 2, 3]})
    row_reordered = pl.DataFrame({"a": [3, 2, 1], "b": [6, 5, 4]})
    both_reordered = pl.DataFrame({"b": [6, 5, 4], "a": [3, 2, 1]})

    hashes = {
        hash_dataframe(table, method=method)
        for table in (original, field_reordered, row_reordered, both_reordered)
    }

    assert len(hashes) == 1


@methods
def test_hash_ignores_list_element_order(method: HashMethod) -> None:
    """List elements hash as a set. `[1, 2]` and `[2, 1]` are the same cell."""
    ordered = pl.DataFrame({"a": [1, 2, 3], "b": [[1, 2], [3, 4], [5, 6]]})
    reordered = pl.DataFrame({"a": [1, 2, 3], "b": [[2, 1], [4, 3], [6, 5]]})

    assert hash_dataframe(ordered, method=method) == hash_dataframe(
        reordered, method=method
    )


# -- content sensitivity --------------------------------------------------------------


@methods
@pytest.mark.parametrize(
    "changed",
    [
        pytest.param(pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 7]}), id="one-value"),
        pytest.param(
            pl.DataFrame({"b": [1, 2, 3], "a": [4, 5, 6]}),
            id="content-swapped-between-columns",
        ),
    ],
)
def test_hash_changes_on_content(method: HashMethod, changed: pl.DataFrame) -> None:
    """Order-invariance must not reach so far it stops noticing a real difference."""
    original = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    assert hash_dataframe(original, method=method) != hash_dataframe(
        changed, method=method
    )


@methods
def test_hash_changes_on_struct(method: HashMethod) -> None:
    """Structs hash by their contents, nested values included, not their shape alone."""
    basic = pl.DataFrame(
        {"id": [1, 2], "meta": [{"name": "Alice", "age": 30}, {"name": "Bob"}]}
    )
    changed = pl.DataFrame(
        {"id": [1, 2], "meta": [{"name": "Alice", "age": 31}, {"name": "Bob"}]}
    )

    assert hash_dataframe(basic, method=method) != hash_dataframe(
        changed, method=method
    )


@methods
def test_hash_binary_columns(method: HashMethod) -> None:
    """Binary is hex-encoded before hashing, so non-UTF-8 bytes survive."""
    table = pl.DataFrame({"a": [1, 2, 3], "b": [b"abc", None, bytes([255, 254, 253])]})

    assert isinstance(hash_dataframe(table, method=method), bytes)


# -- as_sorted_list: hashing columns as a set -----------------------------------------


@methods
def test_sorted_list_off_by_default(method: HashMethod) -> None:
    """By default, `(left, right)` is ordered, so swapping the two changes the hash."""
    original = pl.DataFrame({"left_id": [1, 2, 3], "right_id": [4, 5, 6]})
    swapped = pl.DataFrame({"left_id": [4, 5, 6], "right_id": [1, 2, 3]})

    assert hash_dataframe(original, method=method) != hash_dataframe(
        swapped, method=method
    )


@methods
def test_sorted_list_ignores_id_order(method: HashMethod) -> None:
    """Swapped or row-reordered IDs hash the same. Changed IDs do not."""
    sort_on = ["left_id", "right_id"]
    original = pl.DataFrame(
        {"left_id": [1, 2, 3], "right_id": [4, 5, 6], "score": [0.8, 0.9, 0.7]}
    )
    swapped = pl.DataFrame(
        {"left_id": [4, 5, 6], "right_id": [1, 2, 3], "score": [0.8, 0.9, 0.7]}
    )
    reordered = pl.DataFrame(
        {"left_id": [2, 1, 3], "right_id": [5, 4, 6], "score": [0.9, 0.8, 0.7]}
    )
    changed = pl.DataFrame(
        {"left_id": [1, 2, 3], "right_id": [4, 5, 6], "score": [0.8, 0.9, 0.8]}
    )

    hashes = {
        hash_dataframe(table, method=method, as_sorted_list=sort_on)
        for table in (original, swapped, reordered)
    }
    assert len(hashes) == 1
    assert hash_dataframe(changed, method=method, as_sorted_list=sort_on) not in hashes


@methods
def test_sorted_list_many_columns(method: HashMethod) -> None:
    """Wider than a pair: the same trio in any columns hashes the same."""
    sort_on = ["person_a", "person_b", "person_c"]
    abc = pl.DataFrame(
        {"person_a": [1], "person_b": [4], "person_c": [7], "score": [0.8]}
    )
    cab = pl.DataFrame(
        {"person_a": [7], "person_b": [1], "person_c": [4], "score": [0.8]}
    )

    assert hash_dataframe(abc, method=method, as_sorted_list=sort_on) == hash_dataframe(
        cab, method=method, as_sorted_list=sort_on
    )


@methods
def test_sorted_list_with_nulls(method: HashMethod) -> None:
    """A null is a value in the set, so two frames with the same nulls sorted agree."""
    sort_on = ["left_id", "right_id"]
    a = pl.DataFrame(
        {"left_id": [1, None, 3], "right_id": [None, 5, 6], "score": [0.8, 0.9, 0.7]}
    )
    b = pl.DataFrame(
        {"left_id": [None, 5, 6], "right_id": [1, None, 3], "score": [0.8, 0.9, 0.7]}
    )

    hash_a = hash_dataframe(a, method=method, as_sorted_list=sort_on)
    hash_b = hash_dataframe(b, method=method, as_sorted_list=sort_on)
    assert hash_a == hash_b


# -- empty tables ---------------------------------------------------------------------


@methods
def test_hash_empty_table(method: HashMethod) -> None:
    """An empty table has one stable hash, whatever its columns, unlike a full one.

    This is a behavioural check rather than an assertion on the literal sentinel the
    function returns. That value is an implementation detail. What a store relies on is
    the contract that empty is stable and distinct.
    """
    empty_two_col = pl.DataFrame({"a": [], "b": []})
    empty_one_col = pl.DataFrame({"x": []})
    populated = pl.DataFrame({"a": [1], "b": [2]})

    assert hash_dataframe(empty_two_col, method=method) == hash_dataframe(
        empty_one_col, method=method
    )
    assert hash_dataframe(empty_two_col, method=method) != hash_dataframe(
        populated, method=method
    )
