"""Base class for deduplication methodologies."""

from abc import ABC, abstractmethod
from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field


class Deduper(BaseModel, ABC):
    """A methodology that finds candidate duplicate pairs within one record step.

    A `Model` step calls `prepare()` once, then `dedupe()`, each time it collects. Put
    one-off setup in `prepare()` instead, for example fitting a model over the whole
    dataset, so it doesn't repeat on every call to `dedupe()`. `dedupe()` must return
    a table with `left_id`, `right_id`, and `score` columns. `normalise_model_scores`
    casts that table to `SCHEMA_MODEL_EDGES`.

    Every field is a setting — carried in a document and hashed into the model's
    fingerprint — unless marked `matchlab.resources.FromResources`, in which case it is
    supplied through a model's `model_resources`. Since a fingerprint ignores resources,
    a marked field must be something the methodology reads *through*, never something
    that changes what it scores.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

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
