"""Representation of human judgements."""

from itertools import chain
from typing import Self

from pydantic import BaseModel, Field, model_validator


class Judgement(BaseModel):
    """A user's decision on how one reviewed cluster's leaves split into entities."""

    tag: str | None = None
    shown: list[int] = Field(
        description="Leaf IDs of the whole cluster shown to the user"
    )
    endorsed: list[list[int]] = Field(
        description="Leaf IDs the user grouped together, one list per endorsed entity"
    )

    @model_validator(mode="after")
    def check_no_duplicates(self) -> Self:
        """Reject leaf IDs repeated within `shown`, or within `endorsed`."""
        if len(self.shown) != len(set(self.shown)):
            raise ValueError("One or more leaf IDs were repeated in the shown data")
        concat_ids = list(chain(*self.endorsed))
        if len(concat_ids) != len(set(concat_ids)):
            raise ValueError(
                "One or more leaf IDs were repeated in the endorsement data"
            )

        return self

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        """Check that `endorsed`'s leaves are exactly the ones in `shown`."""
        all_shown = set(self.shown)
        all_endorsed = set(chain(*self.endorsed))
        if all_shown != all_endorsed:
            raise ValueError("Inconsistent leaf IDs between shown and endorsed")

        return self
