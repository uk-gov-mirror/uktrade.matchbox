"""Base class for linkers."""

from abc import ABC, abstractmethod
from typing import ClassVar, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field


class Linker(BaseModel, ABC):
    """A methodology that finds candidate matches between two record steps.

    A `Model` step calls `prepare()` once, then `link()`, each time it collects. Put
    one-off setup in `prepare()` instead, for example fitting a model over both
    datasets, so it doesn't repeat on every call to `link()`. `link()` must return a
    table with `left_id`, `right_id`, and `score` columns. `normalise_model_scores`
    casts that table to `SCHEMA_MODEL_EDGES`.

    Every field is a setting unless marked `matchlab.resources.FromResources`. A
    fingerprint ignores a resource, so a marked field must not change what this scores.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Bump it to retire every artifact the previous body wrote.
    # Left unset, the step re-runs on every collect, and so does everything below it.
    # See `matchlab.core.versioning`.
    version: ClassVar[int | None] = None

    left_id: Literal["id"] = Field(
        default="id", description="The unique ID field in the left data"
    )
    right_id: Literal["id"] = Field(
        default="id", description="The unique ID field in the right data"
    )

    @abstractmethod
    def prepare(self, left: pl.DataFrame, right: pl.DataFrame) -> None:
        """Run once before `link()`, for setup that shouldn't repeat per call."""
        ...

    @abstractmethod
    def link(self, left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
        """Score candidate matches between `left` and `right`."""
        ...
