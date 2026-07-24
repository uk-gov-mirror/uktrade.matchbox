"""Source — the leaf of a plan.

A source reads rows from a warehouse and content-addresses them. It takes no inputs,
so it is where raw data (and therefore non-determinism) enters a plan: its
configuration key includes a hash of the data it read, which is what makes a freshly
constructed `Source` pick up warehouse changes while an existing object memoises its
read.

**The extract/transform is the whole declaration.** Every column it returns is part of
the record, so every column except the key contributes to that record's identity — two
rows are the same leaf when they agree on all of them. There is no second list of
indexed fields to keep in step with the SQL, and no field types to restate: a column you
do not want to affect identity is a column you should not select, and a type you want
pinned is a `cast` in the SELECT.
"""

from __future__ import annotations

import re
import tempfile
from collections.abc import Callable, Generator, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl
import pyarrow.parquet as pq

from matchlab.adapters import Adapter, Fingerprint
from matchlab.core.config import SourceConfig
from matchlab.core.db import QueryReturnClass, QueryReturnType
from matchlab.core.hash import HashMethod, hash_arrow_table, hash_rows
from matchlab.core.logging import logger, profile_time
from matchlab.core.resolution import leaf_id
from matchlab.steps import Step
from matchlab.views import View

if TYPE_CHECKING:
    from matchlab.locations import Location
    from matchlab.models import Model


# A source's name prefixes every column it contributes (`crn` gives `crn_company`),
# and those land in cleaning SQL that sqlglot parses. `crn-x_company` would parse as
# subtraction and `crn.x_company` as table.column, so the name has to be usable at the
# start of an unquoted identifier. Reserved words are fine — the name is only ever a
# prefix, never a bare identifier.
_IDENTIFIER = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class Source(Step):
    """A warehouse table, extracted and content-addressed."""

    kind: ClassVar[str] = "source"

    def __init__(
        self,
        location: Location,
        name: str,
        extract_transform: str,
        key_field: str,
        validate_etl: bool = True,
    ) -> None:
        """Define a source.

        Args:
            location: Where the data lives.
            name: The source's name within the plan. Prefixes every column this
                source contributes, so it must be usable in a SQL identifier.
            extract_transform: SQL producing the rows to index. Every column it
                returns other than the key contributes to record identity.
            key_field: The name of the unique identifier field. Read as a string
                whatever the warehouse returns it as.
            validate_etl: Validate the extract/transform SQL up front.

        Raises:
            ValueError: If the name could not prefix a SQL identifier, or if
                `key_field` is not a column name.
        """
        if not _IDENTIFIER.match(name):
            raise ValueError(
                f"Source name '{name}' can't prefix a column name. A source's name "
                f"qualifies every column it contributes — '{name}_company' here — and "
                "those appear in cleaning SQL, so it must start with a letter or "
                "underscore and contain only letters, digits and underscores."
            )

        super().__init__(name=name)

        if not isinstance(key_field, str):
            raise ValueError(
                f"key_field must be a column name, got {type(key_field).__name__}"
            )

        if validate_etl:
            location.validate_extract_transform(extract_transform)

        self.location = location
        self.extract_transform = extract_transform
        self.key_field = key_field

        # Memoised warehouse read: (extract, hashes). Populated on first fingerprint.
        self._read: tuple[pl.DataFrame, pl.DataFrame] | None = None

    # -- configuration ----------------------------------------------------------------

    @property
    def config(self) -> SourceConfig:
        """The serialisable configuration for this source."""
        return SourceConfig(
            name=self.name,
            location_config=self.location.config,
            extract_transform=self.extract_transform,
            key_field=self.key_field,
        )

    # -- column naming ----------------------------------------------------------------
    #
    # A view built over several sources qualifies every column with the source it came
    # from, so `company` from `crn` becomes `crn_company`. These say what a column will
    # be called once that has happened, which is how you refer to it in a cleaning
    # expression or a model's settings — before anything has been collected.

    @property
    def prefix(self) -> str:
        """The column prefix this source's fields carry once queried."""
        return f"{self.name}_"

    def qualify_field(self, field: str) -> str:
        """Prefix a single field name with this source's name."""
        return self.prefix + field

    def f(self, fields: str | Iterable[str]) -> str | list[str]:
        """Prefix one or more field names with this source's name."""
        if isinstance(fields, str):
            return self.qualify_field(fields)
        return [self.qualify_field(field) for field in fields]

    @property
    def qualified_key(self) -> str:
        """This source's key field, prefixed with the source name."""
        return self.qualify_field(self.key_field)

    @property
    def index_fields(self) -> list[str]:
        """Every column the extract returns except the key, in sorted order.

        Read from the warehouse rather than declared, so it cannot drift from the SQL.
        """
        extract, _ = self._read_warehouse()
        return sorted(column for column in extract.columns if column != self.key_field)

    @property
    def qualified_index_fields(self) -> list[str]:
        """`index_fields`, prefixed with the source name."""
        return [self.qualify_field(field) for field in self.index_fields]

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

        # Keys are always strings, whatever the warehouse returns them as. Every
        # other column is read as the warehouse types it — pin one with a `cast` in
        # the extract/transform if you need it stable across drivers.
        selection = (self.key_field, keys) if keys else None
        yield from self.location.execute(
            extract_transform=self.extract_transform,
            schema_overrides={self.key_field: pl.String},
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

        Every column except the key contributes to the row hash, so two rows are the
        same leaf exactly when the extract returns identical values for both.

        Returns:
            `(extract, hashes)` — the raw rows, and a `hash → keys` index in which
            byte-identical rows share a hash (content-addressed dedup at index time).
        """
        if self._read is not None:
            return self._read

        logger.info("Reading source data", prefix=f"Read {self.name}")
        key = self.key_field

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
                index = sorted(column for column in batch.columns if column != key)
                if not index:
                    raise ValueError(
                        f"Source '{self.name}' returns only its key field. The "
                        "extract/transform must select at least one other column."
                    )
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
            leaf_id(pl.col("hash")).alias("leaf"),
        )

    # -- Step contract ----------------------------------------------------------------

    def _config_key(self) -> bytes:
        """Configuration plus a content hash of the data read.

        Including the data hash is what lets a plan detect that the warehouse changed:
        a new `Source` object re-reads and gets a different key, invalidating
        everything downstream of it.

        Hashing `hashes` alone is sufficient because every non-key column feeds the row
        hash, and `hashes` records which keys carry each one. A change to any selected
        column therefore moves the fingerprint. This was not true while a separate list
        of index fields could be narrower than the extract: a column outside it changed
        the stored extract without touching the fingerprint, so the source cache-hit,
        never re-stored, and downstream views kept reading the stale value.
        """
        _, hashes = self._read_warehouse()
        return super()._config_key() + hash_arrow_table(hashes.to_arrow())

    def _execute(self, adapter: Adapter, fp: Fingerprint) -> None:
        extract, _ = self._read_warehouse()
        adapter.store_source(
            fp=fp,
            name=self.name,
            key_field=self.key_field,
            extract=extract,
            leaves=self.leaves(),
        )

    # -- verbs ------------------------------------------------------------------------

    def view(self, *, cleaning: dict[str, str] | None = None, **kwargs: Any) -> View:
        """Return a view of this source's records.

        `cleaning` is keyword-only so that this reads the same as
        `Resolver.view(source, cleaning=...)`, where the positional slot is taken by
        the sources being read.
        """
        return View(self, cleaning=cleaning, **kwargs)

    def dedupe(
        self,
        model_class: Any,  # noqa: ANN401 - a Deduper subclass
        model_settings: Any,  # noqa: ANN401 - its settings model or a dict
        name: str | None = None,
    ) -> Model:
        """Deduplicate this source's records.

        Shorthand for `self.view().dedupe(...)` — a view is only worth naming when you
        want to clean, group, or read through a resolver.
        """
        return self.view().dedupe(
            model_class=model_class, model_settings=model_settings, name=name
        )

    def link(
        self,
        other: Source | View,
        model_class: Any,  # noqa: ANN401 - a Linker subclass
        model_settings: Any,  # noqa: ANN401 - its settings model or a dict
        name: str | None = None,
    ) -> Model:
        """Link this source to another source or view.

        Shorthand for `self.view().link(...)`. `other` is viewed too if it is a
        source, so neither side needs one.
        """
        return self.view().link(
            other, model_class=model_class, model_settings=model_settings, name=name
        )
