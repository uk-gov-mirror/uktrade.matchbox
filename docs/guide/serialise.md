# Serialise a plan

A plan is Python objects. To run the same plan somewhere else, dump it to a **document**, move the document, and load it there.

A document is JSON. It is derived from a plan, never written by hand.

```python
from pathlib import Path

import matchlab as mb

Path("plan.json").write_text(mb.dump(entities))
```

`dump()` takes the apex of the plan and walks upstream, covering exactly what `collect()` would run. It hands back JSON, so where the document goes is your call: a file, a column, an object store.

## What a document holds

```json
{
  "steps": [
    {
      "kind": "source",
      "spec": {
        "location_class": "RelationalDB",
        "location_settings": {"sql": "select pk, company, town from companies"},
        "name": "crn",
        "key_field": "pk"
      },
      "inputs": [],
      "resources": {"client": "warehouse"}
    },
    {
      "kind": "transform",
      "spec": {
        "transformer_class": "Clean",
        "transformer_settings": {"cleaning": {"name": "lower(crn_company)"}}
      },
      "inputs": [0],
      "resources": {}
    },
    {
      "kind": "model",
      "spec": {
        "model_type": "deduper",
        "model_class": "NaiveDeduper",
        "model_settings": {"id": "id", "unique_fields": ["name"]}
      },
      "inputs": [1],
      "resources": {}
    },
    {
      "kind": "resolver",
      "spec": {"resolver_class": "Components", "resolver_settings": {"thresholds": {}}},
      "inputs": [2],
      "resources": {}
    }
  ]
}
```

Each node holds three things:

* `spec` is the step's own settings, and exactly what its [fingerprint](../glossary.md#fingerprint) hashes.
* `inputs` names the steps it reads.
* `resources` names what it reads *through*, such as a database client.

Nodes refer to each other by [position](../glossary.md#position), the same number `draw()` shows in brackets. Steps have no names, so there is nothing else to refer to them by. A node's inputs always sit earlier in the list than the node itself.

Structural sharing survives. A record step feeding two models is one node referenced twice, not two copies of a subtree.

## Settings and resources

A **setting** is serialisable configuration. It travels in the document and is hashed into the step's fingerprint. A **resource** is an object that can't be serialised, such as a database engine or a dataframe. A document records only the name that resource was given.

That split is what lets a plan move without its credentials.

To dump a plan, name every resource it uses:

```python
from sqlalchemy import create_engine

warehouse = mb.Resource("warehouse", create_engine("postgresql://..."))

crn = mb.read_database(
    "crn", sql="select pk, company from crn", client=warehouse, key_field="pk"
)
dh = mb.read_database(
    "dh", sql="select pk, company from dh", client=warehouse, key_field="pk"
)
```

Sharing a name means sharing the object. Give two sources reading one warehouse the same `Resource`. One name covering two different objects is refused.


A bare engine works for the whole life of the process. Only `dump()` needs the name:

```
ResourceError: Step 0 (source) was given an unnamed resource for 'client', so this
plan cannot be described. Wrap it: client=Resource("some_name", ...).
```

Renaming a resource changes no fingerprint anywhere in the plan. Editing a setting re-runs that step and everything below it. That is the difference the two arguments exist to make.

### Which fields are which

Every field of a location, transformer, deduper, linker or resolver is a setting, unless its class marks it [`FromResources`](../api/plan/resources.md):

```python
# matchlab's own RelationalDB, quoted from source
class RelationalDB(Location):
    sql: SQLQuery  # a setting
    client: FromResources[DBClient]  # a resource
```

Settings go in a step's `*_settings` argument and resources in its `*_resources` argument. `read_database` and `read_dataframe` sort them for you. Build a step directly and you pass both:

```python
mb.Source(
    location_class=mb.RelationalDB,
    name="crn",
    location_settings={"sql": "select pk, company from crn"},
    location_resources={"client": warehouse},
    key_field="pk",
)
```

## Loading

`load()` rebuilds the plan and returns its apex. Nothing is collected. Hand it the JSON:

```python
entities = mb.load(
    Path("plan.json").read_text(),
    resources={"warehouse": create_engine("postgresql://...")},
)
entities.collect()
```

`resources` is keyed by resource name. Miss any out and `load()` says so before rebuilding anything.

## Fingerprints transfer

A rebuilt plan fingerprints identically to the plan that was dumped, given the same data. A store already holding the original's artifacts serves them to the rebuilt plan instead of redoing the work.

The sources decide it. A source's fingerprint folds in a hash of the rows it read. A document carries no data and no hash of any, so that hash is derived on load, by reading the target's own rows.

* Same rows, same fingerprints, and the target store hits cache.
* Different rows, different fingerprints, and the plan re-runs.

## What a document can't carry

* **Resources.** Only the name each one was given. Nothing secret enters a document.
* **Code.** `location_class`, `model_class` and the rest are registry names.
* **Labels.** [Publishing](./build-a-plan.md#publishing) is something you do to a collected result. Load, collect, then publish under whatever label that environment wants.
* **Data.** Rows are read from the target's own warehouse.

## Custom classes

A document names each class by its registered name. The target environment has to register the same classes:

```python
mb.add_location_class(S3Location)
```

Without it, the document still parses and `load()` fails:

```
ValueError: No location class named 'S3Location' is registered. Register it with
`add_location_class` before loading a document that names it.
```

`add_model_class`, `add_transformer_class` and `add_resolver_class` do the same for the other kinds of step. A document is portable across environments, not across codebases.
