"""Representation of human judgements."""

from itertools import chain
from typing import Self

from pydantic import BaseModel, Field, model_validator


class Judgement(BaseModel):
    """User determination on how to group source clusters from a model cluster."""

    tag: str | None = None
    shown: list[int] = Field(
        description="IDs of the source clusters shown to user at once"
    )
    endorsed: list[list[int]] = Field(
        description="""Groups of source cluster IDs that user thinks belong together"""
    )

    @model_validator(mode="after")
    def check_no_duplicates(self) -> Self:
        """Ensure no cluster IDs are repeated in the endorsement."""
        if len(self.shown) != len(set(self.shown)):
            raise ValueError("One or more cluster IDs were repeated in the shown data")
        concat_ids = list(chain(*self.endorsed))
        if len(concat_ids) != len(set(concat_ids)):
            raise ValueError(
                "One or more cluster IDs were repeated in the endorsement data"
            )

        return self

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        """Ensure union of endorsed clusters matches shown cluster."""
        all_shown = set(self.shown)
        all_endorsed = set(chain(*self.endorsed))
        if all_shown != all_endorsed:
            raise ValueError(
                "Inconsistent source cluster IDs between shown and endorsed clusters"
            )

        return self
