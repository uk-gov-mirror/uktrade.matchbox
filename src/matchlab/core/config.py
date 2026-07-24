"""Configuration models — the serialisable description of a plan's steps.

One model per step kind. A step's config is what its `_config_key` hashes, so the rule
for what belongs here is exactly the fingerprint invariant: **a config must carry
everything the step's output depends on, and nothing else.**

That rule explains the one asymmetry. `SourceConfig` carries its own `name`, because a
source's name prefixes every column it contributes and tags its rows in a resolution —
rename it and the output changes. No other step's name reaches its own output, so no
other config records it. Where a name *is* load-bearing to a consumer it appears in the
consumer's config: `ResolverConfig.inputs` records model names because per-model
thresholds are keyed by them.

Configs are flat. They describe a step's own settings, never its inputs' — the plan's
edges live on `Step.upstream`, and a step's fingerprint already folds in its parents'.
Nesting a parent's config inside a child's would duplicate that.
"""

import textwrap
from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class LocationType(StrEnum):
    """Enumeration of location types."""

    RDBMS = "rdbms"


class QueryCombineType(StrEnum):
    """Enumeration of ways to combine multiple rows having the same cluster ID."""

    CONCAT = "concat"
    EXPLODE = "explode"
    SET_AGG = "set_agg"


class ModelType(StrEnum):
    """Enumeration of supported model types."""

    LINKER = "linker"
    DEDUPER = "deduper"


class ResolverType(StrEnum):
    """Enumeration of supported resolver methodology types."""

    COMPONENTS = "components"


class LocationConfig(BaseModel):
    """Metadata for a location."""

    model_config = ConfigDict(frozen=True)

    type: LocationType
    name: str


class SourceConfig(BaseModel):
    """Configuration of a source: where its rows come from, and what keys them.

    There is no separate list of indexed fields. The extract/transform is the single
    declaration of what a source *is* — every column it returns is part of the record,
    and therefore part of that record's identity. A column you do not want to affect
    identity is a column you should not select.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        description=(
            "The source's name within the plan. Part of the config because it is "
            "part of the output: it prefixes every column this source contributes "
            "and tags its rows in a resolution."
        )
    )
    location_config: LocationConfig = Field(
        description=(
            "The location of the source. Used to run the extract/tansform logic."
        ),
    )
    extract_transform: str = Field(
        description=(
            "Logic to extract and transform data from the source. "
            "Language is location dependent."
        )
    )
    key_field: str = Field(
        description=textwrap.dedent("""
            The name of the key field. This is the source's key for unique
            entities, such as a primary key in a relational database.

            Keys are always read as strings, whatever the warehouse returns.

            For example, if the source describes companies, it may have used
            a Companies House number as its key.

            This key is ALWAYS correct. It should be something generated and
            owned by the source being indexed.

            For example, your organisation's CRM ID is a key field within the CRM.

            A CRM ID entered by hand in another dataset shouldn't be used
            as a key field.
        """),
    )


class CleanerConfig(BaseModel):
    """Configuration of a cleaned view over one or more sources."""

    model_config = ConfigDict(frozen=True)

    sources: tuple[str, ...] = Field(
        description="Names of the sources this view reads, in lineage order."
    )
    resolver: str | None = Field(
        default=None,
        description=(
            "The resolver whose clusters this view reads through, if any. Without "
            "one, records are grouped by their source leaves."
        ),
    )
    combine_type: QueryCombineType = Field(
        default=QueryCombineType.CONCAT,
        description="How to combine rows sharing a cluster ID.",
    )
    cleaning: dict[str, str] | None = Field(
        default=None,
        description=(
            "Output column to SQL expression. `None` passes every column through; "
            "an empty dict projects to identifiers only."
        ),
    )


class ModelConfig(BaseModel):
    """Configuration of a deduper or linker."""

    model_config = ConfigDict(frozen=True)

    model_type: ModelType = Field(description="Whether this is a deduper or a linker.")
    model_class: str = Field(
        description="The registered name of the Deduper or Linker subclass."
    )
    model_settings: dict[str, Any] = Field(
        description="That class's settings, dumped to JSON."
    )
    left: str = Field(description="Name of the view being matched.")
    right: str | None = Field(
        default=None, description="Name of the right view, for linkers."
    )


class ResolverConfig(BaseModel):
    """Configuration of a resolver over one or more models."""

    model_config = ConfigDict(frozen=True)

    resolver_class: str = Field(
        description="The registered name of the ResolverMethod subclass."
    )
    resolver_settings: dict[str, Any] = Field(
        description="That class's settings, dumped to JSON."
    )
    inputs: tuple[str, ...] = Field(
        description=(
            "Names of the models whose edges are resolved. Load-bearing: per-model "
            "settings such as score thresholds are keyed by these names."
        )
    )
