"""Pluggable, serialisable reshapings of a frame.

`Select`, `Clean` and `Group` split the old cleaning behaviour so each does one job.
`Select` projects, `Clean` derives, `Group` changes granularity. Register a custom one
with `add_transformer_class`, as `add_model_class` does a deduper.
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
