"""Utilities for hashing data and creating unique identifiers."""

import hashlib
from enum import StrEnum

import polars as pl
import polars.expr as plx
import polars_hash as plh

HASH_FUNC = hashlib.sha256


class HashMethod(StrEnum):
    """Supported hash methods for row hashing."""

    XXH3_128 = "xxh3_128"
    SHA256 = "sha256"


def _process_column_for_hashing(column_name: str, schema_type: pl.DataType) -> plx.Expr:
    r"""Normalise `column_name` to a string column safe to hash.

    Nulls become the sentinel `"\x00"`, so a null hashes differently from an empty
    value. Binary columns hex-encode, structs JSON-encode and lists join on `,`,
    before falling back to a plain string cast.
    """
    if isinstance(schema_type, pl.Binary):
        return (
            pl.col(column_name).fill_null("\x00").bin.encode("hex").alias(column_name)
        )
    elif isinstance(schema_type, pl.Struct):
        return (
            pl.col(column_name)
            .struct.json_encode()
            .fill_null("\x00")
            .alias(column_name)
        )
    elif isinstance(schema_type, pl.List):
        return pl.col(column_name).list.join(",").fill_null("\x00").alias(column_name)
    else:
        return pl.col(column_name).cast(pl.Utf8).fill_null("\x00").alias(column_name)


def hash_rows(
    df: pl.DataFrame, columns: list[str], method: HashMethod = HashMethod.XXH3_128
) -> pl.Series:
    """Hash each row of `df` over `columns`.

    Each row's hash covers both column names and values, joined with the record and
    unit separator symbols (`␞`, `␟`), so a column's name is part of what gets hashed
    alongside its value.
    """
    expr_list = [
        _process_column_for_hashing(column, df.schema[column]) for column in columns
    ]
    df_processed = df.with_columns(expr_list)

    record_separator = "␞"
    unit_separator = "␟"

    str_concatenation: list[pl.Expr] = []
    for c in columns:
        str_concatenation.extend(
            [
                pl.lit(c),  # column name
                pl.lit(unit_separator),
                pl.col(c),  # column value
                pl.lit(record_separator),
            ]
        )

    if method == HashMethod.XXH3_128:
        row_hashes = df_processed.select(
            # Pinned explicitly: later polars-hash versions return a u128 int here,
            # and leaves require binary
            plh.concat_str(str_concatenation)
            .nchash.xxh3_128()
            .cast(pl.Binary)
            .alias("row_hash")
        )
        return row_hashes["row_hash"]
    elif method == HashMethod.SHA256:
        row_hashes = df_processed.select(
            plh.concat_str(str_concatenation)
            .chash.sha2_256()
            .str.decode("hex")
            .alias("row_hash")
        )
        return row_hashes["row_hash"]
    else:
        raise ValueError(f"Unsupported hash method: {method}")


def hash_dataframe(
    df: pl.DataFrame,
    method: HashMethod = HashMethod.XXH3_128,
    as_sorted_list: list[str] | None = None,
) -> bytes:
    """Content-address `df` with a hash invariant to row and column order.

    Pass `as_sorted_list` (2 or more column names, e.g. `["left_id", "right_id"]`) to
    hash those columns as a sorted set instead of individually, so `(1, 2)` and
    `(2, 1)` hash the same. This replaces the named columns with one `sorted_list`
    column.

    Combining `as_sorted_list` with a nullable column can null the whole row's hash
    input, because Polars' `concat_list` returns null when any input is null.

    Raises:
        ValueError: If `as_sorted_list` names fewer than two columns, or a column
            `df` doesn't have.
    """
    if df.height == 0:
        return b"empty_table_hash"

    if as_sorted_list:
        if len(as_sorted_list) < 2:
            raise ValueError(
                "Lists passed to as_sorted_list must contain at least 2 column names"
            )

        missing_cols = [col for col in as_sorted_list if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in dataframe: {missing_cols}")

        df = df.with_columns(
            pl.concat_list(as_sorted_list).list.sort().alias("sorted_list")
        ).drop(as_sorted_list)

    columns: list[str] = sorted(df.columns)
    df = df.select(columns)

    # Explode list columns so each element sorts and hashes on its own. That's what
    # makes the hash invariant to the order of elements within a list field.
    for column in columns:
        if isinstance(df.schema[column], pl.List):
            df = df.explode(column, empty_as_null=True)

    df = df.sort(by=columns)
    row_hashes = hash_rows(df=df, columns=columns, method=method)
    all_hashes: bytes = b"".join(row_hashes.sort().to_list())

    return HASH_FUNC(all_hashes).digest()
