"""Generate data whose answer you already know, then score matching against it.

The point of this package is to make entity resolution *measurable*: it plants entities
you know the answer for, so a test can assert precision and recall against truth rather
than eyeball a fixture.

One job, four beats — **describe** the data (`features`), **generate** it and plant the
answer (`sources`, `linked`), **match** it (`matchers`), **score** the result
(`compare`). `LinkedSources` is the handle for all of it: because it planted the
entities, nothing downstream has to be told the answer.

## Which do I use?

| I want… | Use |
|---|---|
| data whose answer I know | `linked_sources_factory()` → `LinkedSources` |
| one source only | `source_factory()` → `GeneratedSource` |
| rows I wrote by hand | `source_from_tuple()` |
| a plan to test | `linked.dedupe(...)` / `linked.link(..., through=...)` |
| to score a collected plan | `linked.diff_resolution(resolution, *sources)` |
| to score a methodology's edges | `linked.diff_model_edges(edges, left=...)` |
| different columns | `FeatureConfig` + `SuffixRule` / `ReplaceRule` |
| a matcher that already knows the answer | `PerfectDeduper` / `PerfectLinker` |
| to build or bend the answer it matches on | `AnswerKey.from_sources(...)` |
| to compose a comparison by hand | `resolution_to_clusters` + `diff_entities` |

`dedupe()` and `link()` wire a perfect matcher up for you, so the last two rows are
only for when you want a matcher that is deliberately wrong in a particular way.

A testkit exposes what it knows about the *fixture*; anything you do to the *plan* goes
through the node it wraps — `source.source.view()`, `model.model.resolve()`. There are
no forwarding shortcuts, so it is always clear which of the two you are holding.

```python
from matchlab.testkit import linked_sources_factory

linked = linked_sources_factory(
    n_true_entities=10, engine=warehouse
).write_to_location()
resolution = linked.link("crn", "cdms").model.resolve().collect().entities()

identical, report = linked.diff_resolution(resolution, "crn", "cdms")
assert identical, report
```

## Two layers, two questions

Ground truth is expressed twice, in different ID spaces. Neither subsumes the other, and
picking the wrong one is the easiest mistake to make here:

| | Synthetic-ID layer | Value-keyed layer |
|---|---|---|
| Asks | does this *methodology* work? | does the *plan* carry grouping forward? |
| Built from | `GeneratedModel.predicted_clusters` | `AnswerKey` + `Perfect*` |
| Scored with | `diff_model_edges` | `diff_resolution` |
| Needs a warehouse | no — the view read is mocked | yes, and a real `collect()` |
| Example | `test/methodologies/` | `test/plan/test_ground_truth.py` |

The split exists because matchlab derives record identity by content-hashing rows at
collect time, so IDs are unknowable when a fixture is built; anything asserted against a
*collected* plan has to be keyed by row values instead. See `matchers` for the full
reasoning. `dedupe()`/`link()` build both, so either is available from one call.

Names not re-exported below are still importable from their module, but they are
plumbing rather than entry points.
"""

from matchlab.testkit.compare import diff_entities, resolution_to_clusters
from matchlab.testkit.entities import Cluster, TrueEntity
from matchlab.testkit.features import (
    FeatureConfig,
    ReplaceRule,
    SourceParameters,
    SuffixRule,
)
from matchlab.testkit.linked import LinkedSources, linked_sources_factory
from matchlab.testkit.matchers import AnswerKey, PerfectDeduper, PerfectLinker
from matchlab.testkit.models import GeneratedModel
from matchlab.testkit.sources import GeneratedSource, source_factory, source_from_tuple

__all__ = (
    # generate data with a known answer
    "linked_sources_factory",
    "LinkedSources",
    "source_factory",
    "source_from_tuple",
    "GeneratedSource",
    "GeneratedModel",
    # match, when you want something other than a perfect matcher
    "PerfectDeduper",
    "PerfectLinker",
    "AnswerKey",
    # describe what to generate
    "FeatureConfig",
    "SuffixRule",
    "ReplaceRule",
    "SourceParameters",
    # score results against the answer
    "diff_entities",
    "resolution_to_clusters",
    # the vocabulary you get back
    "TrueEntity",
    "Cluster",
)
