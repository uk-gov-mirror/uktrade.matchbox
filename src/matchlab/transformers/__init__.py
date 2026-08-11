"""Pluggable, serialisable reshapings of a frame.

`Select` projects, `Clean` derives, and `Group` changes granularity, each one job.
Register a custom one with `add_transformer_class`, as `add_model_class` does a
deduper.
"""

from matchlab.transformers.base import Transformer
from matchlab.transformers.clean import Clean
from matchlab.transformers.group import Group
from matchlab.transformers.select import Select
from matchlab.transformers.transform import Transform, add_transformer_class

__all__ = (
    "Clean",
    "Group",
    "Select",
    "Transform",
    "Transformer",
    "add_transformer_class",
)
