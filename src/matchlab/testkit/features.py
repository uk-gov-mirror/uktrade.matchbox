"""What to generate: feature declarations and the variations applied to them.

Pure declaration. Nothing here generates or compares anything. `FeatureConfig`
describes one column's values. Its `variations` are what let one true entity produce
more than one form of the same value, which `_generate.generate_rows` turns into rows.
"""

from abc import ABC, abstractmethod
from typing import Generic, Self, TypeVar

import polars as pl
from adbc_driver_manager.dbapi import Connection as AdbcConnection
from faker import Faker
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import Engine

T = TypeVar("T")


class VariationRule(BaseModel, Generic[T], ABC):
    """Abstract base class for variation rules."""

    model_config = ConfigDict(frozen=True)

    @property
    @abstractmethod
    def type(self) -> type[T]:
        """Python type this rule can be applied to."""
        ...

    @abstractmethod
    def apply(self, value: T) -> T:
        """Apply the variation to a value."""
        ...


class SuffixRule(VariationRule[str]):
    """Add a suffix to a value."""

    suffix: str

    @property
    def type(self) -> type[str]:  # noqa: D102
        return str

    def apply(self, value: str) -> str:  # noqa: D102
        return f"{value}{self.suffix}"


class ReplaceRule(VariationRule[str]):
    """Replace occurrences of a string with another."""

    old: str
    new: str

    @property
    def type(self) -> type[str]:  # noqa: D102
        return str

    def apply(self, value: str) -> str:  # noqa: D102
        return value.replace(self.old, self.new)


def infer_data_type(base: str, parameters: tuple | None) -> pl.DataType:
    """Infer the Polars data type a Faker configuration produces.

    Args:
        base: Faker generator type.
        parameters: Parameters for the generator.

    Returns:
        The Polars data type of the generated values.
    """
    generator = Faker()
    value_generator = getattr(generator, base)
    parameters = {} if not parameters else dict(parameters)
    examples = [value_generator(**dict(parameters)) for _ in range(5)]
    series = pl.Series(examples)
    return series.dtype


class FeatureConfig(BaseModel):
    """Configuration for generating a feature with variations."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    base_generator: str
    parameters: tuple | None = Field(
        default=None,
        description=(
            "Parameters for the generator. A tuple of tuples passed to the generator."
        ),
    )
    unique: bool = Field(
        default=True,
        description=(
            "Whether the generator enforces uniqueness in the generated data. "
            "For example, using unique=True with the 'boolean' generator will error "
            "if more than two values are generated."
        ),
    )
    drop_base: bool = Field(
        default=False, description="Whether the base case is dropped."
    )
    variations: tuple[VariationRule, ...] = Field(default_factory=tuple)
    datatype: pl.DataType = Field(
        default_factory=lambda data: infer_data_type(
            data["base_generator"], data["parameters"]
        )
    )

    def add_variations(self, *rule: VariationRule) -> "FeatureConfig":
        """Add a variation rule to the feature."""
        return FeatureConfig(
            name=self.name,
            base_generator=self.base_generator,
            parameters=self.parameters,
            unique=self.unique,
            drop_base=self.drop_base,
            variations=self.variations + tuple(rule),
        )

    @field_validator("name", mode="after")
    @classmethod
    def protected_names(cls: type[Self], value: str) -> str:
        """Ensure name is not a reserved keyword."""
        if value in {"id", "key"}:
            raise ValueError("Feature name cannot be 'id' or 'key'.")
        return value


class SourceParameters(BaseModel):
    """Configuration for generating a source."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    features: tuple[FeatureConfig, ...] = Field(default_factory=tuple)
    name: str
    # Defaults to None rather than an engine, so that every caller ends up with its own
    # warehouse. A `Field(default=create_engine(...))` is evaluated once at class
    # definition, which silently shared one in-memory SQLite database across the whole
    # process, so two sources built without an explicit engine could not be joined.
    engine: Engine | AdbcConnection | None = Field(default=None)
    n_true_entities: int | None = Field(default=None)
    repetition: int = Field(default=0)
