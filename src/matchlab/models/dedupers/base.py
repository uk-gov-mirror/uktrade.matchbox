"""Base class for deduplication methodologies."""

from abc import ABC, abstractmethod
from typing import Literal

import polars as pl
from pydantic import BaseModel, Field


class DeduperSettings(BaseModel):
    """Settings shared by every Deduper methodology."""

    id: Literal["id"] = Field(
        default="id", description="The unique ID field in the data to dedupe"
    )


class Deduper(BaseModel, ABC):
    """A methodology that finds candidate duplicate pairs within one view.

    A `Model` step calls `prepare()` once, then `dedupe()`, each time it collects. Put
    one-off setup in `prepare()` instead, for example fitting a model over the whole
    dataset, so it doesn't repeat on every call to `dedupe()`. `dedupe()` must return
    a frame with `left_id`, `right_id`, and `score` columns. `normalise_model_scores`
    casts that frame to `SCHEMA_MODEL_EDGES`.
    """

    settings: DeduperSettings

    @abstractmethod
    def prepare(self, data: pl.DataFrame) -> None:
        """Run once before `dedupe()`, for setup that shouldn't repeat per call."""
        return

    @abstractmethod
    def dedupe(self, data: pl.DataFrame) -> pl.DataFrame:
        """Score candidate duplicate pairs within `data`."""
        return
