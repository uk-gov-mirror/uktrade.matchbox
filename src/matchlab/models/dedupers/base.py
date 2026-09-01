"""Base class for deduplication methodologies."""

from abc import ABC, abstractmethod
from typing import ClassVar, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field


class Deduper(BaseModel, ABC):
    """A methodology that finds candidate duplicate pairs within one record step.

    A `Model` step calls `prepare()` once, then `dedupe()`, each time it collects. Put
    one-off setup in `prepare()` instead, for example fitting a model over the whole
    dataset, so it doesn't repeat on every call to `dedupe()`. `dedupe()` must return
    a table with `left_id`, `right_id`, and `score` columns. `normalise_model_scores`
    casts that table to `SCHEMA_MODEL_EDGES`.

    Every field is a setting unless marked `matchlab.resources.FromResources`. A
    fingerprint ignores a resource, so a marked field must not change what this scores.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Bump it to retire every artifact the previous body wrote.
    # Left unset, the step re-runs on every collect, and so does everything below it.
    # See `matchlab.core.versioning`.
    version: ClassVar[int | None] = None

    id: Literal["id"] = Field(
        default="id", description="The unique ID field in the data to dedupe"
    )

    @abstractmethod
    def prepare(self, data: pl.DataFrame) -> None:
        """Run once before `dedupe()`, for setup that shouldn't repeat per call."""
        ...

    @abstractmethod
    def dedupe(self, data: pl.DataFrame) -> pl.DataFrame:
        """Score candidate duplicate pairs within `data`."""
        ...
