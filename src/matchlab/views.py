"""View — a queryable, optionally-cleaned view over one or more sources.

`View` is a plan node like any other, and it stores its cleaned table like any other: a
view feeding three models is computed once and read back three times, rather than
rebuilt inside each of them. That is worth the storage because a view is usually a
plan's most shared node — every pairwise link over the same sources reads the same few
views — and because building the frame is the expensive part: a join over the stored
source extracts, then the cleaning SQL.

`data()` reads that same stored table, so inspecting "what does my cleaned data actually
look like?" costs nothing beyond the collection that was going to happen anyway.

What a view does *not* store is `identifiers()` — the `(id, source, key, leaf)` mapping
a downstream resolver needs. That depends only on the sources and resolver a view reads,
never on its cleaning, and the cleaned frame has dropped `source`, `key` and `leaf` by
the time it is stored. It is read back from the source leaves and the upstream
resolution instead, both of which are already stored.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import duckdb
import polars as pl
from sqlglot import expressions, parse_one
from sqlglot import select as sqlglot_select

from matchlab.adapters import Adapter, Fingerprint
from matchlab.core.db import QueryReturnClass, QueryReturnType
from matchlab.core.kinds import StepKind
from matchlab.specs import ViewSpec
from matchlab.steps import Step

if TYPE_CHECKING:
    from matchlab.models import Model
    from matchlab.resolvers import Resolver
    from matchlab.sources import Source

#: What one source's identifiers are read with: `(source_fp, source_name, resolver_fp)`,
#: the arguments of `Adapter.read_identifiers`. Deduplicating these is deduplicating the
#: queries, which is why they travel as a value rather than as a call.
IdentifierRead = tuple["Fingerprint | None", str, "Fingerprint | None"]


class View(Step):
    """A cleaned view over sources, optionally resolved through a resolver."""

    kind: ClassVar[StepKind] = StepKind.VIEW

    def __init__(
        self,
        *sources: Source,
        resolver: Resolver | None = None,
        cleaning: dict[str, str] | None = None,
        group: bool = False,
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

        upstream: tuple[Step, ...] = (*sources, *((resolver,) if resolver else ()))
        super().__init__(upstream=upstream)

    # -- Step contract ----------------------------------------------------------------

    @property
    def spec(self) -> ViewSpec:
        """The serialisable spec for this view."""
        return ViewSpec(cleaning=self.cleaning, group=self.group)

    def _execute(self, adapter: Adapter, fp: Fingerprint) -> None:
        adapter.store_view(fp, self._compute(adapter))

    # -- data -------------------------------------------------------------------------

    @property
    def _identifier_reads(self) -> tuple[IdentifierRead, ...]:
        """The `Adapter.read_identifiers` arguments this view's records come from.

        One per source, naming what is read rather than reading it. Separate from
        `identifiers` because a downstream resolver wants the *set* of these across
        every view feeding it, and two views that clean the same source through the
        same resolver differently read exactly the same rows — so quoting them lets
        the resolver ask once instead of once per consuming model.
        """
        resolver_fp = self.resolver._fp if self.resolver is not None else None
        return tuple((source._fp, source.name, resolver_fp) for source in self.sources)

    def identifiers(self, adapter: Adapter) -> pl.DataFrame:
        """Return `(id, source, key, leaf)` for every record this view reads.

        `id` is the resolver's root cluster when reading through one, otherwise the
        source leaf. This is the *upstream resolution* a downstream resolver needs to
        carry every reachable leaf forward — including records no model matched.
        """
        return pl.concat(
            [adapter.read_identifiers(*read) for read in self._identifier_reads],
            how="vertical",
        )

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
        """Return the cleaned data from the stored table.

        Unconditional, because `collect` runs a step's inputs before the step itself —
        so by the time a consumer asks, this view's `_ensure` has already stored it or
        found it cached. Reading rather than recomputing is what makes a shared view
        cost one computation instead of one per consumer, and it is what lets a plan
        rebuilt in a new process pick up a view stored by an earlier one.
        """
        if self._fp is None:  # pragma: no cover - collect orders upstream first
            raise RuntimeError(
                "This view has not been collected. Call collect() first."
            )
        return adapter.read_view(self._fp)

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
    ) -> Model:
        """Deduplicate this view."""
        from matchlab.models import Model  # noqa: PLC0415 - avoids a cycle

        return Model(left=self, model_class=model_class, model_settings=model_settings)

    def link(
        self,
        other: Source | View,
        model_class: Any,  # noqa: ANN401 - a Linker subclass
        model_settings: Any,  # noqa: ANN401 - its settings model or a dict
    ) -> Model:
        """Link this view to another source or view.

        `other` is viewed for you if it is a source, so the right-hand side of a link
        never needs one unless you want to clean or group it.
        """
        from matchlab.models import Model  # noqa: PLC0415 - avoids a cycle

        right = other if isinstance(other, View) else other.view()
        return Model(
            left=self,
            right=right,
            model_class=model_class,
            model_settings=model_settings,
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
