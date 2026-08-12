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
from matchlab.locations import RelationalDBLocation
from matchlab.models import Model
from matchlab.recordstep import RecordStep
from matchlab.resolvers import Resolver
from matchlab.sources import Source
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
    "Group",
    "Model",
    "PlanDocument",
    "RecordStep",
    "RelationalDBLocation",
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
    "set_default_adapter",
)
