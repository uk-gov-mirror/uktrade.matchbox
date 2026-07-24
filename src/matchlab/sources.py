"""Source — the leaf of a plan.

A source reads rows from a warehouse and content-addresses them. It takes no inputs,
so it is where raw data (and therefore non-determinism) enters a plan: its
configuration key includes a hash of the data it read, which is what makes a freshly
constructed `Source` pick up warehouse changes while an existing object memoises its
read.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Generator, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl
import pyarrow.parquet as pq

from matchlab.adapters import Adapter, Fingerprint
from matchlab.cleaning import Clean
from matchlab.core.config import SourceConfig, SourceField
from matchlab.core.datatypes import DataTypes
from matchlab.core.db import QueryReturnClass, QueryReturnType
from matchlab.core.hash import HashMethod, hash_arrow_table, hash_rows
from matchlab.core.logging import logger, profile_time
from matchlab.core.resolution import leaf_id
from matchlab.steps import Step

if TYPE_CHECKING:
    from matchlab.locations import Location
    from matchlab.models import Model


class Source(Step):
    """A warehouse table, extracted and content-addressed."""

    kind: ClassVar[str] = "source"

    def __init__(
        self,
        location: Location,
        name: str,
        extract_transform: str,
        key_field: str | SourceField,
        index_fields: list[str] | list[SourceField],
        infer_types: bool | None = None,
        validate_etl: bool = True,
    ) -> None:
        """Define a source.

        Args:
            location: Where the data lives.
            name: The source's name within the plan.
            extract_transform: SQL producing the rows to index.
            key_field: The unique identifier field, or a `SourceField` for it.
            index_fields: The fields to match on, or `SourceField`s for them.
            infer_types: Infer field types from the warehouse. Defaults to inferring
                when fields are given as names, and not when they are already
                `SourceField` instances.
            validate_etl: Validate the extract/transform SQL up front.
        """
        super().__init__(name=name)

        if infer_types is None:
            infer_types = isinstance(key_field, str)

        if validate_etl:
            location.validate_extract_transform(extract_transform)

        self.location = location
        self.extract_transform = extract_transform

        if infer_types:
            self._check_field_types(key_field, index_fields, str)
            inferred = location.infer_types(extract_transform)
            self.key_field = SourceField(name=key_field, type=DataTypes.STRING)
            self.index_fields = tuple(
                SourceField(name=field, type=inferred[field]) for field in index_fields
            )
        else:
            self.key_field, self.index_fields = self._check_field_types(
                key_field, index_fields, SourceField
            )

        # Memoised warehouse read: (extract, hashes). Populated on first fingerprint.
        self._read: tuple[pl.DataFrame, pl.DataFrame] | None = None
        # Memoised content hash of the extract, computed alongside the first read.
        self._extract_hash: bytes | None = None

    @staticmethod
    def _check_field_types(
        key_field: str | SourceField,
        index_fields: Iterable[str | SourceField],
        expected: type,
    ) -> tuple[Any, tuple[Any, ...]]:
        if not isinstance(key_field, expected):
            raise ValueError(
                f"Expected {expected.__name__}, got {type(key_field).__name__}"
            )
        fields = tuple(index_fields)
        if not all(isinstance(field, expected) for field in fields):
            raise ValueError(f"All index_fields must be {expected.__name__} instances")
        return key_field, fields

    # -- configuration ----------------------------------------------------------------

    @property
    def config(self) -> SourceConfig:
        """The serialisable configuration for this source."""
        return SourceConfig(
            location_config=self.location.config,
            extract_transform=self.extract_transform,
            key_field=self.key_field,
            index_fields=self.index_fields,
        )

    @property
    def prefix(self) -> str:
        """The column prefix this source's fields carry once queried."""
        return self.config.prefix(self.name)

    @property
    def qualified_key(self) -> str:
        """This source's key field, prefixed with the source name."""
        return self.config.qualified_key(self.name)

    @property
    def qualified_index_fields(self) -> list[str]:
        """This source's index fields, prefixed with the source name."""
        return self.config.qualified_index_fields(self.name)

    def qualify_field(self, field: str) -> str:
        """Prefix a single field name with this source's name."""
        return self.config.qualify_field(self.name, field)

    def f(self, fields: str | Iterable[str]) -> str | list[str]:
        """Prefix one or more field names with this source's name."""
        return self.config.f(self.name, fields)

    # -- warehouse access -------------------------------------------------------------

    def fetch(
        self,
        qualify_names: bool = False,
        batch_size: int | None = None,
        return_type: QueryReturnType = QueryReturnType.POLARS,
        keys: list[str] | None = None,
    ) -> Generator[QueryReturnClass, None, None]:
        """Apply the extract/transform and yield the resulting rows in batches."""
        rename: Callable[[str], str] | None = None
        if qualify_names:

            def rename(column: str) -> str:
                return f"{self.name}_{column}"

        all_fields = (*self.index_fields, self.key_field)
        schema_overrides = {field.name: field.type.to_dtype() for field in all_fields}

        selection = (self.key_field.name, keys) if keys else None
        yield from self.location.execute(
            extract_transform=self.extract_transform,
            schema_overrides=schema_overrides,
            rename=rename,
            batch_size=batch_size,
            return_type=return_type,
            **({"keys": selection} if selection else {}),
        )

    def sample(
        self, n: int = 100, return_type: QueryReturnType = QueryReturnType.POLARS
    ) -> QueryReturnClass:
        """Peek at the first `n` rows without collecting."""
        return next(self.fetch(batch_size=n, return_type=return_type))

    @profile_time(attr="name")
    def _read_warehouse(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Read the source and content-address every row. Memoised.

        Returns:
            `(extract, hashes)` — the raw rows, and a `hash → keys` index in which
            byte-identical rows share a hash (content-addressed dedup at index time).
        """
        if self._read is not None:
            return self._read

        logger.info("Reading source data", prefix=f"Read {self.name}")
        key = self.key_field.name
        index = sorted(field.name for field in self.index_fields)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{self.name}.parquet"
            writer = None
            for batch in self.fetch(return_type=QueryReturnType.ARROW):
                if writer is None:
                    writer = pq.ParquetWriter(path, schema=batch.schema)
                writer.write_table(batch)
            if writer is None:
                raise ValueError(f"Source '{self.name}' returned no rows")
            writer.close()

            frames: list[pl.DataFrame] = []
            hashed: list[pl.DataFrame] = []
            parquet = pq.ParquetFile(path)
            for group in range(parquet.num_row_groups):
                batch = pl.from_arrow(parquet.read_row_group(group))
                if batch[key].is_null().any():
                    raise ValueError(f"Source '{self.name}' has null keys")
                frames.append(batch)
                row_hashes = hash_rows(
                    df=batch, columns=index, method=HashMethod.SHA256
                )
                hashed.append(
                    batch.select(pl.col(key).alias("keys")).with_columns(
                        row_hashes.alias("hash")
                    )
                )

        extract = pl.concat(frames)
        hashes = pl.concat(hashed).group_by("hash").agg(pl.col("keys"))
        self._read = (extract, hashes)
        return self._read

    def leaves(self) -> pl.DataFrame:
        """Return `(key, leaf)` — each source key mapped to its leaf cluster."""
        _, hashes = self._read_warehouse()
        return hashes.explode("keys", empty_as_null=True).select(
            pl.col("keys").alias("key"),
            pl.col("hash").map_elements(leaf_id, return_dtype=pl.UInt64).alias("leaf"),
        )

    # -- Step contract ----------------------------------------------------------------

    def _config_key(self) -> bytes:
        """Configuration plus a content hash of the data read.

        Including the data hash is what lets a plan detect that the warehouse changed:
        a new `Source` object re-reads and gets a different key, invalidating
        everything downstream of it.

        Both halves of the read are hashed, and both are load-bearing:

        * `hashes` content-addresses rows by their **index fields**, which is what
          determines leaf identity and therefore what matches.
        * `extract` is every column the extract/transform selected. Those need not all
          be indexed — you might index on company and postcode but pull town through
          for a cleaning expression or for `view_cluster` — and `Clean` reads the
          stored extract. Hashing only `hashes` would leave a change to a non-indexed
          column invisible to the fingerprint, so the source would cache-hit, never
          re-store, and every downstream view would keep reading the stale column.
        """
        extract, hashes = self._read_warehouse()
        if self._extract_hash is None:
            self._extract_hash = hash_arrow_table(extract.to_arrow())
        payload = json.dumps(
            {"config": self.config.model_dump(mode="json"), "name": self.name},
            sort_keys=True,
        ).encode()
        return payload + hash_arrow_table(hashes.to_arrow()) + self._extract_hash

    def _execute(self, adapter: Adapter, fp: Fingerprint) -> None:
        extract, _ = self._read_warehouse()
        adapter.store_source(
            fp=fp, name=self.name, extract=extract, leaves=self.leaves()
        )

    # -- verbs ------------------------------------------------------------------------

    def clean(self, cleaning: dict[str, str] | None = None, **kwargs: Any) -> Clean:
        """Return a cleaned, queryable view of this source."""
        return Clean(self, cleaning=cleaning, **kwargs)

    def dedupe(self, *args: Any, **kwargs: Any) -> Model:
        """Deduplicate this source. Shorthand for `self.clean().dedupe(...)`."""
        return self.clean().dedupe(*args, **kwargs)

    def link(self, other: Source | Clean, *args: Any, **kwargs: Any) -> Model:
        """Link this source to another source or cleaned view."""
        return self.clean().link(other, *args, **kwargs)
