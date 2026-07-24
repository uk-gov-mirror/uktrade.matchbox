"""The matchlab plan API.

Build a plan from a `Source`, chain verbs, then `collect()`:

```python
from matchlab import Source

deduped = source.clean(...).dedupe(...).resolve()
lookup = deduped.collect().get_matches().as_lookup()
```

Nothing runs until `collect()`, and collecting again only does the work whose plan
or inputs changed.
"""

from matchlab.cleaning import Clean
from matchlab.locations import RelationalDBLocation
from matchlab.models import Model
from matchlab.resolvers import Resolve
from matchlab.sources import Source
from matchlab.steps import Step, default_adapter, gc, set_default_adapter

__all__ = (
    "Clean",
    "Model",
    "RelationalDBLocation",
    "Resolve",
    "Source",
    "Step",
    "default_adapter",
    "gc",
    "set_default_adapter",
)
