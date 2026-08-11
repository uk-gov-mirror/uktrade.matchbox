"""The kinds of step a plan is made of."""

from enum import StrEnum


class StepKind(StrEnum):
    """Enumeration of the kinds of step a plan is made of."""

    SOURCE = "source"
    TRANSFORM = "transform"
    MODEL = "model"
    RESOLVER = "resolver"
