"""Frame — the records a model matches over, and the verbs that build a plan.

A `Frame` is any step whose artifact is a table of records carrying an `id`: a `Source`
read on its own (`id` is the record's leaf), a `Resolved` read of sources through a
resolver (`id` is the entity root), or a `Transform` that reshapes one of those. A
`Model` matches over a `Frame`, and every `Frame` chains the same verbs — `select`,
`clean`, `group`, `transform`, `dedupe`, `link` — so a source, a resolved read and a
transform all read the same.

`Frame` is not user-facing. Users hold a `Source`, a `Resolved` or a `Transform` and
call verbs on it. `Frame` is where those verbs and the shared `identifiers()`/`data()`
live, so each concrete kind only supplies how it materialises (`_read_cache`) and which
source rows it stands for (`_identifier_reads`).

What a frame does *not* store is `identifiers()`, the `(id, source, key, leaf)` mapping
a downstream resolver needs. That depends only on the sources and resolver a frame
reads, never on any reshaping, so it is read back from the source leaves and the
upstream resolver output, both already stored.
"""

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

from matchlab.adapters import Adapter, Fingerprint
from matchlab.core.dataframes import (
    DataFrameClass,
    DataFrameType,
    qualify,
    to_dataframe,
)
from matchlab.core.kinds import StepKind
from matchlab.specs import ResolvedSpec
from matchlab.steps import Step

if TYPE_CHECKING:
    # Each of these modules imports this one.
    from matchlab.models import Model
    from matchlab.resolvers import Resolver
    from matchlab.sources import Source
    from matchlab.transformers import Transform, Transformer

# What one source's identifiers are read with: `(source_fp, source_name, resolver_fp)`,
# the arguments of `Adapter.read_identifiers`. Deduplicating these is deduplicating the
# queries, which is why they travel as a value rather than as a call.
IdentifierRead = tuple[Fingerprint | None, str, Fingerprint | None]


def build_frame(
    adapter: Adapter,
    sources: "tuple[Source, ...]",
    resolver: "Resolver | None",
) -> pl.DataFrame:
    """Assemble the `id` + qualified-columns frame from stored extracts and identifiers.

    Every source's extract is prefixed with the source name (`company` → `crn_company`)
    and joined to its identifiers, so each row gains the `id` it belongs to: the
    record's leaf when read directly, the entity root when read through `resolver`.
    Several sources are concatenated diagonally, each row carrying its own source's
    columns and nulls for the rest. This is what both `Source` and `Resolved` read.
    """
    resolver_fp = resolver._fp if resolver is not None else None
    identifiers = pl.concat(
        [
            adapter.read_identifiers(source._fp, source.name, resolver_fp)
            for source in sources
        ],
        how="vertical",
    )

    extracts = [
        adapter.read_source_extract(source._fp)
        .select(pl.all().name.prefix(qualify(source.name)))
        .with_columns(pl.lit(source.name).alias("source"))
        .rename({source.qualified_key: "key"})
        for source in sources
    ]

    return (
        pl.concat(extracts, how="diagonal")
        .join(identifiers.drop("leaf"), how="inner", on=("source", "key"))
        .drop("source", "key")
    )


class Frame(Step):
    """A step whose artifact is a table of records a model matches over."""

    # -- Frame contract ---------------------------------------------------------------

    @abstractmethod
    def _read_cache(self, adapter: Adapter) -> pl.DataFrame:
        """Return this frame's records from wherever they are materialised."""
        ...

    @property
    @abstractmethod
    def _identifier_reads(self) -> tuple[IdentifierRead, ...]:
        """The `Adapter.read_identifiers` arguments this frame's records come from.

        One per source, naming what is read rather than reading it. A downstream
        resolver wants the *set* of these across every frame feeding it, since two
        frames that reshape the same source through the same resolver differently read
        exactly the same rows. Quoting them lets the resolver ask once.
        """
        ...

    def identifiers(self, adapter: Adapter) -> pl.DataFrame:
        """Return `(id, source, key, leaf)` for every record this frame reads.

        `id` is the resolver's entity root when reading through one, otherwise the
        source leaf. This is the upstream resolver output a downstream resolver needs to
        carry every reachable leaf forward, including records no model matched.
        """
        return pl.concat(
            [adapter.read_identifiers(*read) for read in self._identifier_reads],
            how="vertical",
        )

    def data(self, return_type: DataFrameType = DataFrameType.POLARS) -> DataFrameClass:
        """Return this frame's records, collecting the plan first if needed."""
        if not self.is_collected:
            self.collect()
        return to_dataframe(self._read_cache(self._require_adapter()), return_type)

    # -- verbs ------------------------------------------------------------------------

    def transform(
        self,
        transformer: "Transformer | type[Transformer] | str",
        transformer_settings: dict | None = None,
    ) -> "Transform":
        """Reshape this frame with a transformer."""
        from matchlab.transformers import Transform  # noqa: PLC0415 - avoids a cycle

        return Transform(self, transformer, transformer_settings)

    def select(self, *columns: str) -> "Transform":
        """Keep only the named columns, plus `id`."""
        from matchlab.transformers import Select  # noqa: PLC0415 - avoids a cycle

        return self.transform(Select(*columns))

    def clean(self, cleaning: dict[str, str]) -> "Transform":
        """Derive columns with DuckDB SQL, keeping the rest."""
        from matchlab.transformers import Clean  # noqa: PLC0415 - avoids a cycle

        return self.transform(Clean(cleaning=cleaning))

    def group(self, aggregates: dict[str, str]) -> "Transform":
        """Collapse each `id` to one row using aggregate SQL."""
        from matchlab.transformers import Group  # noqa: PLC0415 - avoids a cycle

        return self.transform(Group(aggregates=aggregates))

    def dedupe(
        self,
        model_class: Any,  # noqa: ANN401 - a Deduper subclass or its registered name
        model_settings: Any,  # noqa: ANN401 - its settings model or a dict
    ) -> "Model":
        """Deduplicate this frame."""
        from matchlab.models import Model  # noqa: PLC0415 - avoids a cycle

        return Model(left=self, model_class=model_class, model_settings=model_settings)

    def link(
        self,
        other: "Frame",
        model_class: Any,  # noqa: ANN401 - a Linker subclass or its registered name
        model_settings: Any,  # noqa: ANN401 - its settings model or a dict
    ) -> "Model":
        """Link this frame to another. A `Source` is a frame, needing no wrapping."""
        from matchlab.models import Model  # noqa: PLC0415 - avoids a cycle

        return Model(
            left=self,
            right=other,
            model_class=model_class,
            model_settings=model_settings,
        )


class Resolved(Frame):
    """Sources read *through* a resolver, so `id` is the entity root, not the leaf."""

    kind: ClassVar[StepKind] = StepKind.RESOLVED

    def __init__(self, *sources: "Source", resolver: "Resolver") -> None:
        """Define a through-resolver read.

        Args:
            *sources: The sources to read. At least one.
            resolver: The resolver to read them through, so records that resolved to one
                entity share an `id`.

        Raises:
            ValueError: If no sources are given.
        """
        if not sources:
            raise ValueError("A resolved read needs at least one source")

        self._sources = sources
        self.resolver = resolver
        super().__init__(upstream=(*sources, resolver))

    # -- Step contract ----------------------------------------------------------------

    @property
    def sources(self) -> "tuple[Source, ...]":
        """The sources this reads, in the order given."""
        return self._sources

    @property
    def spec(self) -> ResolvedSpec:
        """The serialisable spec. Field-less: its inputs are its identity."""
        return ResolvedSpec()

    def _execute(self, adapter: Adapter, fp: Fingerprint) -> None:
        adapter.store_frame(
            fp, self.kind, build_frame(adapter, self._sources, self.resolver)
        )

    # -- Frame contract ---------------------------------------------------------------

    def _read_cache(self, adapter: Adapter) -> pl.DataFrame:
        if self._fp is None:  # collect orders upstream first
            raise RuntimeError(
                "This resolved read has not been collected. Call collect() first."
            )
        return adapter.read_frame(self._fp)

    @property
    def _identifier_reads(self) -> tuple[IdentifierRead, ...]:
        resolver_fp = self.resolver._fp
        return tuple((source._fp, source.name, resolver_fp) for source in self._sources)
