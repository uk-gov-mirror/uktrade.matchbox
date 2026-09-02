"""Source — the leaf of a plan.

A source reads rows from a `Location` and content-addresses them. It takes no inputs,
so it is where raw data, and therefore non-determinism, enters a plan. Its spec key
includes a hash of the data it read, which is what makes a freshly constructed `Source`
pick up changes at the origin while an existing object memoises its read.
"""

import tempfile
from collections.abc import Callable, Generator, Iterable
from pathlib import Path
from typing import Any, ClassVar

import polars as pl
import pyarrow.parquet as pq

from matchlab.core.dataframes import (
    DataFrameClass,
    DataFrameType,
    qualify,
    validate_col_prefix,
)
from matchlab.core.hash import HashMethod, hash_dataframe, hash_rows
from matchlab.core.kinds import StepKind
from matchlab.core.logging import logger
from matchlab.core.resolver_output import leaf_id
from matchlab.core.sql import SQLQuery
from matchlab.recordstep import IdentifierRead, RecordStep, build_record_step
from matchlab.resources import Resource
from matchlab.sources.base import Location
from matchlab.sources.dataframe import DataFrame
from matchlab.sources.relational import DBClient, RelationalDB
from matchlab.specs import SourceSpec
from matchlab.steps import Step
from matchlab.stores import Fingerprint, Store

_LOCATION_CLASSES: dict[str, "type[Location]"] = {}


def add_location_class(location_class: "type[Location]") -> None:
    """Register a custom location so it can be named in a plan document."""
    if not issubclass(location_class, Location):
        raise ValueError("The argument is not a subclass of Location.")
    _LOCATION_CLASSES[location_class.__name__] = location_class


def resolve_location_class(location_class: "str | type[Location]") -> "type[Location]":
    """Return a location class, looking a name up in the registry.

    How `Source` accepts either the class or its registered name, and how
    `matchlab.document` rebuilds the locations a plan reads.

    Raises:
        ValueError: If no location class of that name is registered here.
    """
    if not isinstance(location_class, str):
        return location_class
    if location_class not in _LOCATION_CLASSES:
        raise ValueError(
            f"No location class named '{location_class}' is registered. Register it "
            "with `add_location_class` before loading a document that names it."
        )
    return _LOCATION_CLASSES[location_class]


for _builtin in (RelationalDB, DataFrame):
    add_location_class(_builtin)


class Source(RecordStep):
    """A location's rows, read and content-addressed.

    A source is a `RecordStep`. Read directly, its records are its extract, with `id`
    set to each row's content-addressed leaf, so a model can match over it with no
    intervening step.
    """

    kind: ClassVar[StepKind] = StepKind.SOURCE

    _READ_ONLY: ClassVar[frozenset[str]] = RecordStep._READ_ONLY | frozenset(
        {
            "location",
            "location_class",
            "location_settings",
            "location_resources",
            "name",
            "key_field",
        }
    )

    # Settled at construction. Declared, rather than only assigned, so a reader and a
    # type checker can both see what a source holds: `_set` writes them past
    # `__setattr__`, which is invisible to static analysis.
    location: Location
    location_class: type[Location]
    location_settings: dict[str, Any]
    location_resources: dict[str, Resource]
    name: str
    key_field: str

    def __init__(
        self,
        location_class: "type[Location] | str",
        name: str,
        location_settings: dict[str, Any] | None = None,
        location_resources: dict[str, Any] | None = None,
        key_field: str = "id",
    ) -> None:
        """Define a source.

        Args:
            location_class: A `Location` subclass, or its registered name.
            name: The source's name within the plan. Prefixes every column this
                source contributes, so it must be usable in a SQL identifier.
            location_settings: That location's configuration — the query, for a
                `RelationalDB`. Serialisable, carried in a document, and hashed into
                this source's fingerprint.
            location_resources: Resources the location needs that cannot be serialised,
                keyed by field name, such as
                `{"client": Resource("warehouse", engine)}`. See `matchlab.resources`.
            key_field: The name of the unique identifier field. Read as a string
                whatever the location returns it as.

        Raises:
            ValueError: If the name could not prefix a SQL identifier, or if
                `key_field` is not a column name.
            ResourceError: If a field was passed in the wrong one of
                `location_settings` and `location_resources`, or if one resource name
                covers two different objects.
        """
        validate_col_prefix(name)

        if not isinstance(key_field, str):
            raise ValueError(
                f"key_field must be a column name, got {type(key_field).__name__}"
            )

        resolved_class = resolve_location_class(location_class)
        built, settings, resources = self._build_methodology(
            resolved_class, location_settings, location_resources
        )

        super().__init__()

        # Memoised read: (extract, hashes). Populated on first fingerprint.
        self._read: tuple[pl.DataFrame, pl.DataFrame] | None = None

        # A source's name is part of the outputs
        self._set(
            location=built,
            location_class=resolved_class,
            location_settings=settings,
            location_resources=resources,
            name=name,
            key_field=key_field,
        )

        # A leaf checks too: one location can put two fields under one resource name.
        self._check_names()

    @property
    def parents(self) -> tuple[Step, ...]:
        """A source is a leaf in a plan."""
        return ()

    @property
    def _claimed_name(self) -> str:
        """A source's name is part of its output, so no two in one plan may share it."""
        return self.name

    def _methodology_class(self) -> type | None:
        """A source is keyed by a hash of the rows it read, not by any code."""
        return None

    # -- spec -------------------------------------------------------------------------

    def __str__(self) -> str:
        """A source is drawn with its name, since it has one."""
        return f"{self.kind} '{self.name}'"

    @property
    def spec(self) -> SourceSpec:
        """The serialisable spec for this source."""
        return SourceSpec(
            name=self.name,
            key_field=self.key_field,
            location_class=self.location_class.__name__,
            location_settings=self.location_settings,
        )

    # -- column naming ----------------------------------------------------------------
    #
    # A record step built over several sources qualifies every column with the source
    # it came from, so `company` from `crn` becomes `crn_company`. These say what a
    # column will be called once that has happened, which is how you refer to it in a
    # cleaning expression or a model's settings, before anything has been collected.

    def f(self, fields: str | Iterable[str]) -> str | list[str]:
        """Prefix one or more field names with this source's name."""
        if isinstance(fields, str):
            return qualify(self.name, fields)
        return [qualify(self.name, field) for field in fields]

    @property
    def qualified_key(self) -> str:
        """This source's key field, prefixed with the source name."""
        return qualify(self.name, self.key_field)

    @property
    def index_fields(self) -> list[str]:
        """Every column the extract returns except the key, in sorted order.

        Read from the location rather than declared, so it cannot drift from what the
        location actually returns.
        """
        extract, _ = self._read_origin()
        return sorted(column for column in extract.columns if column != self.key_field)

    # -- origin access ----------------------------------------------------------------

    def fetch(
        self,
        qualify_names: bool = False,
        batch_size: int | None = None,
        return_type: DataFrameType = DataFrameType.POLARS,
    ) -> Generator[DataFrameClass, None, None]:
        """Read from the location and yield the resulting rows in batches."""
        rename: Callable[[str], str] | None = None
        if qualify_names:

            def rename(column: str) -> str:
                return qualify(self.name, column)

        # Keys are always strings, whatever the location returns them as. Every other
        # column is read as the location types it.
        yield from self.location.read(
            schema_overrides={self.key_field: pl.String()},
            rename=rename,
            batch_size=batch_size,
            return_type=return_type,
        )

    def sample(
        self, n: int = 100, return_type: DataFrameType = DataFrameType.POLARS
    ) -> DataFrameClass:
        """Peek at the first `n` rows without collecting."""
        return next(self.fetch(batch_size=n, return_type=return_type))

    def _read_origin(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Read the source and content-address every row. Memoised.

        Every column except the key contributes to the row hash, so two rows are the
        same leaf exactly when the extract returns identical values for both. Identity
        is content, not the key. See `matchlab.core.resolver_output.leaf_id` for why
        that is the correct anchor for evaluation rather than a storage trick.

        Returns:
            `(extract, hashes)`, the raw rows, and a `hash → keys` index in which
            byte-identical rows share a hash, so records a judge cannot tell apart share
            a leaf.
        """
        if self._read is not None:
            return self._read

        logger.info(f"Reading from {self.location}", prefix=self.name)
        key = self.key_field

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{self.name}.parquet"
            writer = None
            for batch in self.fetch(return_type=DataFrameType.ARROW):
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
                if key not in batch.columns:
                    raise ValueError(self._missing_key(batch.columns))
                if batch[key].is_null().any():
                    raise ValueError(f"Source '{self.name}' has null keys")
                frames.append(batch)
                index = sorted(column for column in batch.columns if column != key)
                if not index:
                    raise ValueError(
                        f"Source '{self.name}' returns only its key field. Its "
                        "location must return at least one other column."
                    )
                # XXH3_128 rather than SHA-256. `leaf_id`
                # slices the row hash to 8 bytes, so a cryptographic digest's strength
                # is discarded before it identifies anything, and the 128 bits that
                # survive into `_spec_key` are ample for telling rows of one table
                # apart. It is ~7x faster, on a path every collect pays, because
                # `_spec_key` re-reads and re-hashes the origin even when the
                # artifact is cached.
                row_hashes = hash_rows(
                    df=batch, columns=index, method=HashMethod.XXH3_128
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

    def _missing_key(self, columns: list[str]) -> str:
        """Explain that the key column is absent, and where the name came from."""
        return (
            f"Source '{self.name}' has no column '{self.key_field}', so there is "
            f"nothing to key its records by. It returned: {', '.join(columns)}. "
            "`key_field` defaults to 'id'; pass the column that uniquely identifies a "
            "record in this source."
        )

    def leaves(self) -> pl.DataFrame:
        """Return `(key, leaf)`, each source key mapped to its leaf cluster."""
        _, hashes = self._read_origin()
        return hashes.explode("keys", empty_as_null=True).select(
            pl.col("keys").alias("key"),
            leaf_id(pl.col("hash")).alias("leaf"),
        )

    # -- Step contract ----------------------------------------------------------------

    def _spec_key(self) -> bytes:
        """The spec plus a content hash of the data read.

        Including the data hash is what lets a plan detect that the origin changed.
        A new `Source` object re-reads and gets a different key, invalidating
        everything downstream of it.

        Hashing `hashes` alone is sufficient because every non-key column feeds the row
        hash, and `hashes` records which keys carry each one. A change to any returned
        column therefore moves the fingerprint. This was not true while a separate list
        of index fields could be narrower than the extract. A column outside it changed
        the stored extract without touching the fingerprint, so the source cache-hit,
        never re-stored, and downstream record steps kept reading the stale value.
        """
        _, hashes = self._read_origin()
        return super()._spec_key() + hash_dataframe(hashes)

    def _execute(self, store: Store, fp: Fingerprint) -> None:
        extract, _ = self._read_origin()
        store.store_source(
            fp=fp,
            key_field=self.key_field,
            extract=extract,
            leaves=self.leaves(),
        )

    # -- RecordStep contract -------------------------------------------------------

    def _read_cache(self, store: Store) -> pl.DataFrame:
        """Return this source's records with each row's leaf as `id`.

        Derived from the extract and leaves stored by `_execute`, a cheap join rather
        than a separately cached artifact. Reading a source on its own means `id` is
        the leaf, so there is no resolver in the read.
        """
        return build_record_step(store, (self,), None)

    @property
    def _identifier_reads(self) -> tuple[IdentifierRead, ...]:
        """Read directly, so `id` is the leaf and no resolver enters the read."""
        return ((self._fp, self.name, None),)


# -- shorthands -----------------------------------------------------------------------
#
# `Source` takes a class plus two dicts, which is what makes every step kind the same
# shape and what a document round-trips through. These are the forms to write by hand:
# one call, named arguments, nothing to assemble.


def read_database(
    name: str,
    *,
    sql: SQLQuery,
    client: DBClient | Resource[DBClient],
    key_field: str = "id",
) -> Source:
    """Read a source from a relational database.

    Args:
        name: The source's name within the plan.
        sql: The query producing the rows. Every column other than the key contributes
            to record identity. Run as written, so trust it: see `RelationalDB`.
        client: A SQLAlchemy engine or ADBC connection, or a `Resource` naming one.
            Name it to dump the plan; share the same `Resource` between sources
            reading one warehouse.
        key_field: The unique identifier column.

    Returns:
        The source, uncollected.
    """
    return Source(
        location_class=RelationalDB,
        name=name,
        location_settings={"sql": sql},
        location_resources={"client": client},
        key_field=key_field,
    )


def read_dataframe(
    name: str,
    *,
    df: DataFrameClass | Resource[DataFrameClass],
    key_field: str = "id",
) -> Source:
    """Read a source from a dataframe already in memory.

    Args:
        name: The source's name within the plan.
        df: A polars, pandas or arrow frame, or a `Resource` naming one. Every column
            other than the key contributes to record identity, so shape the frame
            before handing it over.
        key_field: The unique identifier column.

    Returns:
        The source, uncollected.
    """
    return Source(
        location_class=DataFrame,
        name=name,
        location_resources={"df": df},
        key_field=key_field,
    )
