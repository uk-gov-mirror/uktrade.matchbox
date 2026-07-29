"""Utilities for hashing data and creating unique identifiers."""

import hashlib
from enum import StrEnum

import polars as pl
import polars.expr as plx
import polars_hash as plh
import pyarrow as pa

HASH_FUNC = hashlib.sha256


class HashMethod(StrEnum):
    """Supported hash methods for row hashing."""

    XXH3_128 = "xxh3_128"
    SHA256 = "sha256"


def _process_column_for_hashing(column_name: str, schema_type: pl.DataType) -> plx.Expr:
    """Process a column for hashing based on its type.

    Args:
        column_name: The column name
        schema_type: The polars schema type of the column

    Returns:
        A polars expression for processing the column
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
    """Hash all rows in a dataframe.

    Args:
        df: The DataFrame to hash rows from
        columns: The column names to include in the hash
        method: The hash method to use

    Returns:
        List of row hashes as bytes
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
            plh.concat_str(str_concatenation).nchash.xxh3_128().alias("row_hash")
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


def hash_arrow_table(
    table: pa.Table,
    method: HashMethod = HashMethod.XXH3_128,
    as_sorted_list: list[str] | None = None,
) -> bytes:
    """Computes a content hash of an Arrow table invariant to row and field order.

    This is used to content-address an Arrow table for caching.

    Args:
        table: The pyarrow Table to hash
        method: The method to use for hashing rows (XXH3_128 or SHA256)
        as_sorted_list: Optional list of column names to hash as a sorted list.
            For example, ["left_id", "right_id"] will create a "sorted_list"
            column and drop the original columns to ensure (1,2) and (2,1)
            hash to the same value. Works with 2 or more columns.

            Note: if list columns are combined with a column that's nullable,
            list + null value returns null. See Polars' concat_list documentation
            for more details.

    Returns:
        Bytes representing the content hash of the table
    """
    df = pl.from_arrow(table)

    if df.height == 0:
        return b"empty_table_hash"

    # Apply normalisation if specified
    if as_sorted_list:
        if len(as_sorted_list) < 2:
            raise ValueError(
                "Lists passed to as_sorted_list must contain at least 2 column names"
            )

        # Check that all columns exist
        missing_cols = [col for col in as_sorted_list if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in dataframe: {missing_cols}")

        # Create normalised group and drop original columns
        df = df.with_columns(
            pl.concat_list(as_sorted_list).list.sort().alias("sorted_list")
        ).drop(as_sorted_list)

    columns: list[str] = sorted(df.columns)
    df = df.select(columns)

    # Explode list fields
    for column in columns:
        if isinstance(df.schema[column], pl.List):
            df = df.explode(column, empty_as_null=True)

    df = df.sort(by=columns)
    row_hashes = hash_rows(df=df, columns=columns, method=method)
    all_hashes: bytes = b"".join(row_hashes.sort().to_list())

    return HASH_FUNC(all_hashes).digest()
