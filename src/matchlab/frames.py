"""Frame, the records a model matches over, and the verbs that build a plan.

A `Frame` is a step whose artifact is a table of records carrying an `id`. `Source`,
`Resolver` and `Transform` are the three kinds. On a `Source`, `id` is the leaf. On a
`Resolver`, `id` is the entity root. Every `Frame` chains the same verbs (`select`,
`clean`, `group`, `transform`, `dedupe`, `link`). A `Model` reads a `Frame` and yields
edges, not records.

`Frame` is not user-facing. Users hold a `Source`, a `Resolver` or a `Transform` and
call verbs on it. Each concrete kind supplies only how it materialises (`_read_cache`)
and which source rows it stands for (`_identifier_reads`).

A frame does not store `identifiers()`, the `(id, source, key, leaf)` mapping a
downstream resolver needs. Reshaping cannot change it, so it is read back from the
source leaves and the upstream resolver output instead.
"""

from abc import abstractmethod
from typing import TYPE_CHECKING

import polars as pl

from matchlab.adapters import Adapter, Fingerprint
from matchlab.core.dataframes import (
    DataFrameClass,
    DataFrameType,
    qualify,
    to_dataframe,
)
from matchlab.steps import Step

if TYPE_CHECKING:
    # Each of these modules imports this one. Only for annotations, so importing the
    # real classes (rather than typing them `Any`) costs nothing at runtime.
    from matchlab.models import Model
    from matchlab.models.dedupers.base import Deduper, DeduperSettings
    from matchlab.models.linkers.base import Linker, LinkerSettings
    from matchlab.resolvers import Resolver
    from matchlab.sources import Source
    from matchlab.transformers import Transform, Transformer

# What one source's identifiers are read with, `(source_fp, source_name, resolver_fp)`,
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
    and joined to its identifiers, so each row gains the `id` it belongs to. That is the
    record's leaf when read directly, or the entity root when read through `resolver`.
    Several sources are concatenated diagonally, each row carrying its own source's
    columns and nulls for the rest. This is what both `Source` and `Resolver` read.
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

        One per source, naming what is read rather than reading it. Two frames built
        from the same source and resolver read identical rows, even if they then
        reshape those rows differently. A downstream resolver wants the *set* of these
        reads across every frame feeding it, so naming them lets it ask once instead of
        once per frame.
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
        model_class: "type[Deduper] | str",
        model_settings: "DeduperSettings | dict",
    ) -> "Model":
        """Deduplicate this frame."""
        from matchlab.models import Model  # noqa: PLC0415 - avoids a cycle

        return Model(left=self, model_class=model_class, model_settings=model_settings)

    def link(
        self,
        other: "Frame",
        model_class: "type[Linker] | str",
        model_settings: "LinkerSettings | dict",
    ) -> "Model":
        """Link this frame to another. A `Source` is a frame, needing no wrapping."""
        from matchlab.models import Model  # noqa: PLC0415 - avoids a cycle

        return Model(
            left=self,
            right=other,
            model_class=model_class,
            model_settings=model_settings,
        )
