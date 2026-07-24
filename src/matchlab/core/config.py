"""Configuration models — the serialisable description of a plan's steps."""

import re
import textwrap
from collections.abc import Iterable
from enum import StrEnum
from typing import Annotated, Self, TypeAlias

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from matchlab.core.datatypes import DataTypes
from matchlab.core.exceptions import NameValidationError


def validate_name(value: str) -> str:
    """Validate a plan step name.

    Args:
        value: The name to validate

    Returns:
        The validated name

    Raises:
        NameValidationError: If the name contains invalid characters
    """
    pattern = r"^[a-zA-Z0-9_.-]+$"
    if not re.match(pattern, value):
        raise NameValidationError(
            f"Name '{value}' is invalid. It can only include "
            "alphanumeric characters, underscores, dots or hyphens."
        )
    return value


Name: TypeAlias = Annotated[
    str,
    StringConstraints(
        pattern=r"^[a-zA-Z0-9_.-]+$",
        min_length=1,
        strip_whitespace=True,
    ),
    AfterValidator(validate_name),
    Field(
        description=(
            "Valid name for a plan step. "
            "Must contain only alphanumeric characters, underscores, dots, or hyphens."
        ),
        examples=["my-dataset", "user_data.v2", "experiment_001"],
        json_schema_extra={
            "pattern": r"^[a-zA-Z0-9_.-]+$",
        },
    ),
]


class LocationType(StrEnum):
    """Enumeration of location types."""

    RDBMS = "rdbms"


SourceStepName: TypeAlias = Name
"""Type alias for source step names."""

ModelStepName: TypeAlias = Name
"""Type alias for model step names."""

ResolverStepName: TypeAlias = Name
"""Type alias for resolver step names."""

StepName: TypeAlias = SourceStepName | ModelStepName | ResolverStepName
"""Type alias for any step names."""


class LocationConfig(BaseModel):
    """Metadata for a location."""

    model_config = ConfigDict(frozen=True)

    type: LocationType
    name: str


class SourceField(BaseModel):
    """A field in a source that can be indexed."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        description=(
            "The name of the field in the source after the "
            "extract/transform logic has been applied."
        )
    )
    type: DataTypes = Field(
        description="The cached field type. Used to ensure a stable hash.",
    )


class SourceConfig(BaseModel):
    """Configuration of a source that can, or has been, indexed in the backend.

    They are foundational processes on top of which linking and deduplication models can
        build new steps.
    """

    model_config = ConfigDict(frozen=True)

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
    # Fields can to be set at creation, or initialised with `.default_columns()`
    key_field: SourceField = Field(
        description=textwrap.dedent("""
            The key field. This is the source's key for unique
            entities, such as a primary key in a relational database.

            Keys must ALWAYS be a string.

            For example, if the source describes companies, it may have used
            a Companies House number as its key.

            This key is ALWAYS correct. It should be something generated and
            owned by the source being indexed.
            
            For example, your organisation's CRM ID is a key field within the CRM.
            
            A CRM ID entered by hand in another dataset shouldn't be used 
            as a key field.
        """),
    )
    index_fields: tuple[SourceField, ...] = Field(
        default=None,
        description=textwrap.dedent(
            """
            The fields to index in this source, after the extract/transform logic 
            has been applied. 

            This is usually set manually, and should map onto the columns that the
            extract/transform logic returns.
            """
        ),
    )

    @model_validator(mode="after")
    def validate_key_field(self) -> Self:
        """Ensure that the key field is a string and not in the index fields."""
        if self.key_field in self.index_fields:
            raise ValueError("Key field must not be in the index fields. ")

        if self.key_field.type != DataTypes.STRING:
            raise ValueError("Key field must have string type.")

        return self

    @property
    def dependencies(self) -> list[StepName]:
        """Local execution prerequisites.

        While this can contain information about graph topology, it should only be used
        to check validity, never to reconstruct it.
        """
        return []

    @property
    def parents(self) -> list[StepName]:
        """Direct DAG edges to this node."""
        return []

    def prefix(self, name: str) -> str:
        """Get the prefix for the source.

        Args:
            name: The name of the source.

        Returns:
            The prefix string (name + "_").
        """
        return name + "_"

    def qualified_key(self, name: str) -> str:
        """Get the qualified key for the source.

        Args:
            name: The name of the source.

        Returns:
            The qualified key field name.
        """
        return self.qualify_field(name, self.key_field.name)

    def qualified_index_fields(self, name: str) -> list[str]:
        """Get the qualified index fields for the source.

        Args:
            name: The name of the source.

        Returns:
            List of qualified index field names.
        """
        return [self.qualify_field(name, field.name) for field in self.index_fields]

    def qualify_field(self, name: str, field: str) -> str:
        """Qualify field names with the source name.

        Args:
            name: The name of the source.
            field: The field name to qualify.

        Returns:
            A single qualified field.
        """
        return self.prefix(name) + field

    def f(self, name: str, fields: str | Iterable[str]) -> str | list[str]:
        """Qualify one or more field names with the source name.

        Args:
            name: The name of the source.
            fields: The field name to qualify, or a list of field names.

        Returns:
            A single qualified field, or a list of qualified field names.
        """
        if isinstance(fields, str):
            return self.qualify_field(name, fields)
        return [self.qualify_field(name, field_name) for field_name in fields]


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
