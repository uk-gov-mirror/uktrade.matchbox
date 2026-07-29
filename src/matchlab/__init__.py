"""The matchlab plan API.

Build a plan from a `Source`, chain verbs, then `collect()`:

```python
from matchlab import Source

deduped = source.view(...).dedupe(...).resolve()
lookup = deduped.collect().get_lookup()
```

Nothing runs until `collect()`, and collecting again only does the work whose plan
or inputs changed.
"""

from matchlab.document import PlanDocument, dump, load
from matchlab.locations import RelationalDBLocation
from matchlab.models import Model
from matchlab.resolvers import Resolver
from matchlab.sources import Source
from matchlab.steps import Step, default_adapter, set_default_adapter
from matchlab.views import View

__all__ = (
    "Model",
    "PlanDocument",
    "RelationalDBLocation",
    "Resolver",
    "Source",
    "Step",
    "View",
    "default_adapter",
    "dump",
    "load",
    "set_default_adapter",
)
