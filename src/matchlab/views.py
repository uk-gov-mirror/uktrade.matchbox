"""View — a queryable, optionally-cleaned view over one or more sources.

`View` is a plan node like any other, but it is **fused by default**: collecting a
plan does not materialise it, the model that consumes it just builds its frame
inline. Collecting a `View` *directly* materialises it, after which downstream steps
read the stored table instead of recomputing — the equivalent of Polars' `.cache()`,
and the way to inspect "what does my cleaned data actually look like?".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Self

import duckdb
import polars as pl
from sqlglot import expressions, parse_one
from sqlglot import select as sqlglot_select

from matchlab.adapters import Adapter, Fingerprint
from matchlab.core.config import ViewConfig
from matchlab.core.db import QueryReturnClass, QueryReturnType
from matchlab.steps import Step

if TYPE_CHECKING:
    from matchlab.models import Model
    from matchlab.resolvers import Resolver
    from matchlab.sources import Source


class View(Step):
    """A cleaned view over sources, optionally resolved through a resolver."""

    kind: ClassVar[str] = "clean"
    stores: ClassVar[bool] = False  # fused unless collected directly

    def __init__(
        self,
        *sources: Source,
        resolver: Resolver | None = None,
        cleaning: dict[str, str] | None = None,
        group: bool = False,
        name: str | None = None,
    ) -> None:
        """Define a view.

        Args:
            *sources: The sources to read.
            resolver: Read the sources *through* this resolver, so `id` is its root
                cluster rather than a source leaf.
            cleaning: Output column name → SQL expression over the source fields.
                `None` passes every column through; `{}` is a real projection that
                selects nothing, leaving only `id`.
            group: Collapse each `id` to a single row. Every cleaning expression must
                then be an aggregate — `any_value(crn_company)` where records agree,
                `list(distinct crn_town)` where they don't. Useful when reading
                through a resolver, where several records share an `id`.
            name: Optional plan name; derived from the sources when omitted.

        Raises:
            ValueError: If no sources are given, or `group` is set without cleaning
                expressions to aggregate with.
        """
        if not sources:
            raise ValueError("A view needs at least one source")

        if group and cleaning is None:
            raise ValueError(
                "group=True needs cleaning expressions: grouping collapses several "
                "records into one row, so each column has to say how it combines. "
                "Pass aggregates such as {'name': 'any_value(crn_company)'}."
            )

        self.sources = sources
        self.resolver = resolver
        self.cleaning = cleaning
        self.group = group

        # The resolver is part of the derived name: reading a source directly and
        # reading it *through* a resolver are different views, and two steps in one
        # plan may not share a name.
        stem = "_".join(source.name for source in sources)
        suffix = f".{resolver.name}" if resolver else ""
        upstream: tuple[Step, ...] = (*sources, *((resolver,) if resolver else ()))
        super().__init__(name=name or f"clean_{stem}{suffix}", upstream=upstream)

    # -- Step contract ----------------------------------------------------------------

    @property
    def config(self) -> ViewConfig:
        """The serialisable configuration for this view."""
        return ViewConfig(
            sources=tuple(source.name for source in self.sources),
            resolver=self.resolver.name if self.resolver else None,
            cleaning=self.cleaning,
            group=self.group,
        )

    def _execute(self, adapter: Adapter, fp: Fingerprint) -> None:
        adapter.store_clean(fp, self._compute(adapter))

    def collect(self, adapter: Adapter | None = None) -> Self:
        """Materialise this cleaned view (and its inputs) rather than fusing it."""
        self.stores = True
        return super().collect(adapter)

    # -- data -------------------------------------------------------------------------

    def identifiers(self, adapter: Adapter) -> pl.DataFrame:
        """Return `(id, source, key, leaf)` for every record this view reads.

        `id` is the resolver's root cluster when reading through one, otherwise the
        source leaf. This is the *upstream resolution* a downstream resolver needs to
        carry every reachable leaf forward — including records no model matched.
        """
        frames: list[pl.DataFrame] = []
        for source in self.sources:
            if self.resolver is not None:
                resolution = adapter.read_resolver(self.resolver._fp)
                frames.append(
                    resolution.filter(pl.col("source") == source.name).select(
                        pl.col("root").alias("id"),
                        pl.lit(source.name).alias("source"),
                        pl.col("key"),
                        pl.col("leaf"),
                    )
                )
            else:
                frames.append(
                    adapter.read_source_leaves(source._fp).select(
                        pl.col("leaf").alias("id"),
                        pl.lit(source.name).alias("source"),
                        pl.col("key"),
                        pl.col("leaf"),
                    )
                )
        return pl.concat(frames, how="vertical")

    def _compute(self, adapter: Adapter) -> pl.DataFrame:
        """Build the cleaned frame from stored source extracts and identifiers."""
        identifiers = self.identifiers(adapter)

        extracts = [
            adapter.read_source_extract(source._fp)
            .select(pl.all().name.prefix(f"{source.name}_"))
            .with_columns(pl.lit(source.name).alias("source"))
            .rename({source.qualified_key: "key"})
            for source in self.sources
        ]

        frame = (
            pl.concat(extracts, how="diagonal")
            .join(identifiers.drop("leaf"), how="inner", on=("source", "key"))
            .drop("source", "key")
        )

        return _apply_cleaning(frame, self.cleaning, group=self.group)

    def _frame(self, adapter: Adapter) -> pl.DataFrame:
        """Return the cleaned data, reading the stored table when materialised."""
        if self.stores and self._fp is not None and adapter.has(self._fp):
            return adapter.read_clean(self._fp)
        return self._compute(adapter)

    def data(
        self, return_type: QueryReturnType = QueryReturnType.POLARS
    ) -> QueryReturnClass:
        """Return this view's data, collecting the plan first if needed."""
        if not self.is_collected:
            self.collect()
        return _convert(self._frame(self._require_adapter()), return_type)

    # -- verbs ------------------------------------------------------------------------

    def dedupe(
        self,
        model_class: Any,  # noqa: ANN401 - a Deduper subclass
        model_settings: Any,  # noqa: ANN401 - its settings model or a dict
        name: str | None = None,
    ) -> Model:
        """Deduplicate this view."""
        from matchlab.models import Model  # noqa: PLC0415 - avoids a cycle

        return Model(
            left=self,
            model_class=model_class,
            model_settings=model_settings,
            name=name,
        )

    def link(
        self,
        other: Source | View,
        model_class: Any,  # noqa: ANN401 - a Linker subclass
        model_settings: Any,  # noqa: ANN401 - its settings model or a dict
        name: str | None = None,
    ) -> Model:
        """Link this view to another source or cleaned view."""
        from matchlab.models import Model  # noqa: PLC0415 - avoids a cycle

        right = other if isinstance(other, View) else other.view()
        return Model(
            left=self,
            right=right,
            model_class=model_class,
            model_settings=model_settings,
            name=name,
        )


def _apply_cleaning(
    data: pl.DataFrame, cleaning: dict[str, str] | None, group: bool = False
) -> pl.DataFrame:
    """Apply cleaning SQL, passing `id` through automatically.

    `None` means "no cleaning" and passes the frame through untouched. An *empty*
    dict is a real projection that selects no columns, so it yields just `id` — the
    two are deliberately not the same.

    With `group`, the projection becomes `GROUP BY id`, so every expression must be
    an aggregate. DuckDB reports a non-aggregate itself, naming the column.
    """
    if cleaning is None:
        return data

    identifier = expressions.Alias(
        this=expressions.Column(this=expressions.Identifier(this="id")),
        alias=expressions.Identifier(this="id"),
    )
    projection: list[expressions.Expression] = [identifier]
    for alias, sql in cleaning.items():
        projection.append(expressions.alias_(parse_one(sql, dialect="duckdb"), alias))

    query = sqlglot_select(*projection, dialect="duckdb").from_("data")
    if group:
        query = query.group_by("id")

    with duckdb.connect(":memory:") as connection:
        connection.register("data", data)
        return connection.execute(query.sql(dialect="duckdb")).pl()


def _convert(data: pl.DataFrame, return_type: QueryReturnType) -> QueryReturnClass:
    match return_type:
        case QueryReturnType.POLARS:
            return data
        case QueryReturnType.PANDAS:
            return data.to_pandas()
        case QueryReturnType.ARROW:
            return data.to_arrow()
        case _:
            raise ValueError(f"Return type {return_type} is invalid")
