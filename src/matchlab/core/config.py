"""Configuration models — the serialisable description of a plan's steps.

One model per step kind. A step's config is what its `_config_key` hashes, so the rule
for what belongs here is exactly the fingerprint invariant: **a config must carry
everything the step's output depends on, and nothing else.**

That rule explains the one asymmetry. `SourceConfig` carries its own `name`, because a
source's name prefixes every column it contributes and tags its rows in a resolution —
rename it and the output changes. No other step's name reaches its own output, so no
other config records one, its own or an input's. A setting that has to point at an
input points at its **position** (`ComponentsSettings.thresholds`), which is identity
rather than nomenclature: renaming a step moves no fingerprint anywhere.

Configs describe **settings, not edges**. A config is a step's own configuration and
never a description of its inputs — the plan's edges live on `Step.upstream`, and a
step's fingerprint already folds in its parents'. Recording an input here would be
recorded twice, and a rename would then invalidate a subtree that produces identical
bytes.

That is also why a config is *not* the serialisation format. Serialising a plan needs
the edges; identifying a step must exclude them. `matchlab.document` keeps the two
apart: `PlanDocument` carries the nodes and the edges between them, and each node's
config stays exactly what the fingerprint hashes.
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


class ViewConfig(BaseModel):
    """Configuration of a view: the shape it gives the records a model matches over.

    *Which* records it reads is not here — that is an edge, and edges live on
    `Step.upstream` and in `PlanDocument`. A view over one source and a view over
    another are told apart by their inputs' fingerprints, which `_fingerprint` already
    folds in, in order.
    """

    model_config = ConfigDict(frozen=True)

    cleaning: dict[str, str] | None = Field(
        default=None,
        description=(
            "Output column to SQL expression. `None` passes every column through; "
            "an empty dict projects to `id` only."
        ),
    )
    group: bool = Field(
        default=False,
        description=(
            "Collapse each `id` to one row, so the cleaning expressions run as "
            "aggregates. Changes the grain of what a model sees."
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


class ResolverConfig(BaseModel):
    """Configuration of a resolver over one or more models."""

    model_config = ConfigDict(frozen=True)

    resolver_class: str = Field(
        description="The registered name of the ResolverMethod subclass."
    )
    resolver_settings: dict[str, Any] = Field(
        description=(
            "That class's settings, dumped to JSON. Settings that point at one of the "
            "resolver's inputs key by the input's position, so no name appears here."
        )
    )
