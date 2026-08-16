"""The matchlab plan API.

Build a plan from a `Source`, chain verbs, then `collect()`:

```python
from matchlab import Source

deduped = source.clean(...).dedupe(...).resolve()
lookup = deduped.collect().get_lookup()
```

Nothing runs until `collect()`, and collecting again only does the work whose plan
or inputs changed.
"""

from importlib.metadata import version

from matchlab.document import PlanDocument, dump, load
from matchlab.locations import DataFrame, RelationalDB
from matchlab.models import Model
from matchlab.recordstep import RecordStep
from matchlab.resolvers import Resolver
from matchlab.resources import Resource
from matchlab.sources import Source, read_db, read_df
from matchlab.steps import Step, default_adapter, set_default_adapter
from matchlab.transformers import (
    Clean,
    Group,
    Select,
    Transform,
    Transformer,
    add_transformer_class,
)

__version__ = version("matchlab")

__all__ = (
    "Clean",
    "DataFrame",
    "Group",
    "Model",
    "Resource",
    "PlanDocument",
    "RecordStep",
    "RelationalDB",
    "Resolver",
    "Select",
    "Source",
    "Step",
    "Transform",
    "Transformer",
    "__version__",
    "add_transformer_class",
    "default_adapter",
    "dump",
    "load",
    "read_db",
    "read_df",
    "set_default_adapter",
)
