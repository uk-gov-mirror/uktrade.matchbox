# matchlab

**A local-first library for building, running and evaluating entity resolution
pipelines.**

Record matching is a chore. matchlab makes it a pipeline you can build, run, query and
measure — on your machine, against your warehouse, with nothing to deploy.

```python
from matchlab import Source
from matchlab.models.dedupers import NaiveDeduper

companies = Source(
    location=warehouse,
    name="crn",
    extract_transform="select pk, company, town from companies",
    key_field="pk",
)

entities = (
    companies.clean({"name": "lower(crn_company)"})
    .dedupe(model_class=NaiveDeduper, model_settings={"unique_fields": ["name"]})
    .resolve()
    .collect()
)

entities.lookup_key(from_source="crn", to_sources=["dh"], key="a1")
```

Read the [full documentation](https://uktrade.github.io/matchlab/).

## What it does

* **A lazy plan.** `Source(...).clean(...).dedupe(...).resolve()` builds a tree of steps.
  Nothing runs until you `collect()`.
* **Content-addressed caching.** Re-collecting an unchanged plan does no work. Adding a
  step runs only that step.
* **Materialised resolution.** A collected resolver writes a complete
  `(root, leaf, key, source)` table, so lookups are reads, not re-derivations.
* **Measurement as a first-class job.** Sample clusters, record judgements, score
  precision and recall, and compare methodologies on equal terms.
* **Your data stays put.** matchlab indexes what's in your warehouse; it doesn't copy it.

## What it doesn't do

No server, no accounts, no permissions, nothing to deploy. If you need a shared,
governed matching service, matchlab is not that.

## Installation

```shell
pip install matchlab
```

## Coming from Matchbox?

matchlab is the successor to `matchbox-db`, with the server removed and the client API
rebuilt. It's a hard break — see the
[migration guide](https://uktrade.github.io/matchlab/migration/matchbox-to-matchlab/).

## Development

See our full development guide and coding standards on our
[contribution guide](https://uktrade.github.io/matchlab/contributing/).
