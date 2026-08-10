"""Content-addressing: the hash a fingerprint is built from.

The contract `hash_arrow_table` promises is order-invariance — a table hashes by what it
contains, not the order rows, columns, or list elements happen to arrive in — while any
real change to content changes the hash. `as_sorted_list` extends that to a set of
columns, so `(1, 2)` and `(2, 1)` count as the same pair. Both hash methods must honour
the same contract, so every test runs over both.
"""

import polars as pl
import pyarrow as pa
import pytest

from matchlab.core.hash import HashMethod, hash_arrow_table, hash_rows

methods = pytest.mark.parametrize(
    "method",
    [
        pytest.param(HashMethod.SHA256, id="sha256"),
        pytest.param(HashMethod.XXH3_128, id="xxh3_128"),
    ],
)


# -- row hashing ----------------------------------------------------------------------


@methods
def test_hash_rows_handles_every_column_type(method: HashMethod) -> None:
    """One hash per row over the full spread of dtypes a source can present.

    The dtype assertions guard the fixture: they are what says this really exercises the
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
def test_field_and_row_order_do_not_change_the_hash(method: HashMethod) -> None:
    """The same content in any column or row order is the same table to the hash."""
    original = pa.Table.from_pydict({"a": [1, 2, 3], "b": [4, 5, 6]})
    field_reordered = pa.Table.from_pydict({"b": [4, 5, 6], "a": [1, 2, 3]})
    row_reordered = pa.Table.from_pydict({"a": [3, 2, 1], "b": [6, 5, 4]})
    both_reordered = pa.Table.from_pydict({"b": [6, 5, 4], "a": [3, 2, 1]})

    hashes = {
        hash_arrow_table(table, method=method)
        for table in (original, field_reordered, row_reordered, both_reordered)
    }

    assert len(hashes) == 1


@methods
def test_order_within_a_list_field_does_not_change_the_hash(method: HashMethod) -> None:
    """List elements hash as a set: `[1, 2]` and `[2, 1]` are the same cell."""
    ordered = pa.Table.from_pydict({"a": [1, 2, 3], "b": [[1, 2], [3, 4], [5, 6]]})
    reordered = pa.Table.from_pydict({"a": [1, 2, 3], "b": [[2, 1], [4, 3], [6, 5]]})

    assert hash_arrow_table(ordered, method=method) == hash_arrow_table(
        reordered, method=method
    )


# -- content sensitivity --------------------------------------------------------------


@methods
@pytest.mark.parametrize(
    "changed",
    [
        pytest.param(
            pa.Table.from_pydict({"a": [1, 2, 3], "b": [4, 5, 7]}), id="one-value"
        ),
        pytest.param(
            pa.Table.from_pydict({"b": [1, 2, 3], "a": [4, 5, 6]}),
            id="content-swapped-between-columns",
        ),
    ],
)
def test_changed_content_changes_the_hash(
    method: HashMethod, changed: pa.Table
) -> None:
    """Order-invariance must not reach so far it stops noticing a real difference."""
    original = pa.Table.from_pydict({"a": [1, 2, 3], "b": [4, 5, 6]})

    assert hash_arrow_table(original, method=method) != hash_arrow_table(
        changed, method=method
    )


@methods
def test_changed_struct_content_changes_the_hash(method: HashMethod) -> None:
    """Structs hash by their contents, nested values included, not their shape alone."""
    basic = pa.Table.from_pydict(
        {"id": [1, 2], "meta": [{"name": "Alice", "age": 30}, {"name": "Bob"}]}
    )
    changed = pa.Table.from_pydict(
        {"id": [1, 2], "meta": [{"name": "Alice", "age": 31}, {"name": "Bob"}]}
    )

    assert hash_arrow_table(basic, method=method) != hash_arrow_table(
        changed, method=method
    )


@methods
def test_binary_columns_hash_including_non_utf8_bytes(method: HashMethod) -> None:
    """Binary is hex-encoded before hashing, so non-UTF-8 bytes survive."""
    table = pa.Table.from_pydict(
        {"a": [1, 2, 3], "b": [b"abc", None, bytes([255, 254, 253])]}
    )

    assert isinstance(hash_arrow_table(table, method=method), bytes)


# -- as_sorted_list: hashing columns as a set -----------------------------------------


@methods
def test_column_order_matters_without_as_sorted_list(method: HashMethod) -> None:
    """The default: `(left, right)` is ordered, so swapping the two changes the hash."""
    original = pa.Table.from_pydict({"left_id": [1, 2, 3], "right_id": [4, 5, 6]})
    swapped = pa.Table.from_pydict({"left_id": [4, 5, 6], "right_id": [1, 2, 3]})

    assert hash_arrow_table(original, method=method) != hash_arrow_table(
        swapped, method=method
    )


@methods
def test_as_sorted_list_makes_id_order_irrelevant(method: HashMethod) -> None:
    """Swapped or row-reordered IDs hash the same; changed IDs do not."""
    sort_on = ["left_id", "right_id"]
    original = pa.Table.from_pydict(
        {"left_id": [1, 2, 3], "right_id": [4, 5, 6], "score": [0.8, 0.9, 0.7]}
    )
    swapped = pa.Table.from_pydict(
        {"left_id": [4, 5, 6], "right_id": [1, 2, 3], "score": [0.8, 0.9, 0.7]}
    )
    reordered = pa.Table.from_pydict(
        {"left_id": [2, 1, 3], "right_id": [5, 4, 6], "score": [0.9, 0.8, 0.7]}
    )
    changed = pa.Table.from_pydict(
        {"left_id": [1, 2, 3], "right_id": [4, 5, 6], "score": [0.8, 0.9, 0.8]}
    )

    hashes = {
        hash_arrow_table(table, method=method, as_sorted_list=sort_on)
        for table in (original, swapped, reordered)
    }
    assert len(hashes) == 1
    assert (
        hash_arrow_table(changed, method=method, as_sorted_list=sort_on) not in hashes
    )


@methods
def test_as_sorted_list_spans_more_than_two_columns(method: HashMethod) -> None:
    """Wider than a pair: the same trio in any columns hashes the same."""
    sort_on = ["person_a", "person_b", "person_c"]
    abc = pa.Table.from_pydict(
        {"person_a": [1], "person_b": [4], "person_c": [7], "score": [0.8]}
    )
    cab = pa.Table.from_pydict(
        {"person_a": [7], "person_b": [1], "person_c": [4], "score": [0.8]}
    )

    assert hash_arrow_table(
        abc, method=method, as_sorted_list=sort_on
    ) == hash_arrow_table(cab, method=method, as_sorted_list=sort_on)


@methods
def test_as_sorted_list_treats_nulls_as_content(method: HashMethod) -> None:
    """A null is a value in the set, so two frames with the same nulls sorted agree."""
    sort_on = ["left_id", "right_id"]
    a = pa.Table.from_pydict(
        {"left_id": [1, None, 3], "right_id": [None, 5, 6], "score": [0.8, 0.9, 0.7]}
    )
    b = pa.Table.from_pydict(
        {"left_id": [None, 5, 6], "right_id": [1, None, 3], "score": [0.8, 0.9, 0.7]}
    )

    hash_a = hash_arrow_table(a, method=method, as_sorted_list=sort_on)
    hash_b = hash_arrow_table(b, method=method, as_sorted_list=sort_on)
    assert hash_a == hash_b


# -- empty tables ---------------------------------------------------------------------


@methods
def test_empty_table_hashes_consistently(method: HashMethod) -> None:
    """An empty table has one stable hash, whatever its columns, unlike a full one.

    A behavioural check in place of asserting the literal sentinel the function returns:
    the value is an implementation detail, but "empty is stable and distinct" is the
    contract a store relies on.
    """
    empty_two_col = pa.Table.from_pydict({"a": [], "b": []})
    empty_one_col = pa.Table.from_pydict({"x": []})
    populated = pa.Table.from_pydict({"a": [1], "b": [2]})

    assert hash_arrow_table(empty_two_col, method=method) == hash_arrow_table(
        empty_one_col, method=method
    )
    assert hash_arrow_table(empty_two_col, method=method) != hash_arrow_table(
        populated, method=method
    )
