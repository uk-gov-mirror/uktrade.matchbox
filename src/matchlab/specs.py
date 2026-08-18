"""Step specs — what a fingerprint hashes.

One model per step kind, and `Step._spec_key` hashes it to address that step's artifact.
What belongs in a spec is therefore decided entirely by the fingerprint invariant.
**It must carry everything the step's output depends on, and nothing else.** Omit
something that changes the output, and `collect` hands back a stale artifact without
re-running. Include something that doesn't, and the step re-runs for nothing, along
with everything below it.

That rules out two things. **Edges**, because `_fingerprint` already folds in the
parents' fingerprints. Describing an input here would record it twice, and a rename
would then invalidate a subtree producing identical bytes. **Names** are also excluded,
with one exception: `SourceSpec.name` is in the output it prefixes. A setting that must
point at an input points at its position instead (`ComponentsSettings.thresholds`).

A spec is therefore not the serialisation format. Rebuilding a plan needs the edges
that identifying a step must exclude. `matchlab.document` carries them alongside it.
"""

import textwrap
from enum import StrEnum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


class ModelType(StrEnum):
    """Whether a model is a deduper or a linker."""

    LINKER = "linker"
    DEDUPER = "deduper"


class SourceSpec(BaseModel):
    """Specification of a source: its name, its key, and the location it reads.

    There is no separate list of indexed fields. What the location returns is the single
    declaration of what a source *is*. Every column is part of the record, and therefore
    part of that record's identity. A column you do not want to affect identity is a
    column that should not come back.
    """

    model_config = ConfigDict(frozen=True)

    location_class: str = Field(
        description="The registered name of the Location subclass to read through."
    )
    location_settings: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "That location's configuration, dumped to JSON — the query, for a "
            "`RelationalDB`. Never its resources, which a document names separately "
            "and a fingerprint never sees."
        ),
    )
    name: str = Field(
        description=(
            "The source's name within the plan. Part of the spec because it is "
            "part of the output: it prefixes every column this source contributes "
            "and tags its rows in a resolver's output."
        )
    )
    key_field: str = Field(
        description=textwrap.dedent("""
            The name of the key field. This is the source's key for unique
            entities, such as a primary key in a relational database.

            Keys are always read as strings, whatever the location returns.

            For example, if the source describes companies, it may have used
            a Companies House number as its key.

            This key is ALWAYS correct. It should be something generated and
            owned by the source being indexed.

            For example, your organisation's CRM ID is a key field within the CRM.

            A CRM ID entered by hand in another dataset shouldn't be used
            as a key field.
        """),
    )


class TransformSpec(BaseModel):
    """Specification of a transform, naming the transformer that reshapes the step."""

    model_config = ConfigDict(frozen=True)

    transformer_class: str = Field(
        description="The registered name of the Transformer subclass."
    )
    transformer_settings: dict[str, Any] = Field(
        description="That transformer's configuration, dumped to JSON."
    )


class ModelSpec(BaseModel):
    """Specification of a deduper or linker."""

    model_config = ConfigDict(frozen=True)

    model_type: ModelType = Field(description="Whether this is a deduper or a linker.")
    model_class: str = Field(
        description="The registered name of the Deduper or Linker subclass."
    )
    model_settings: dict[str, Any] = Field(
        description="That class's settings, dumped to JSON."
    )


class ResolverSpec(BaseModel):
    """Specification of a resolver over one or more models."""

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
