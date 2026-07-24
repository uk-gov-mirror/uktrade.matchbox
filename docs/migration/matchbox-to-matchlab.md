# Matchbox to matchlab

matchlab is Matchbox with the server removed and the client API rebuilt around a lazy
plan. This is a hard break: a new package name, a new import root, and a different way
of expressing a pipeline.

There is no compatibility shim. `matchbox-db` receives no further releases.

## Why

The server existed mainly because you needed it to run a DAG. In practice the
collaboration features it enabled went unused — judgements weren't written back, teams
didn't share collections, and nobody built on each other's DAGs. What it did cost was
real: a large codebase, a worse interface, slower pipelines, and no way for analysts
outside government to adopt it at all.

Removing it deleted more than half the code and made the pipeline faster, because
resolution is now computed once when you collect rather than re-derived on every query.

## Install

```bash
pip uninstall matchbox-db
pip install matchlab
```

## Imports

| Matchbox | matchlab |
|---|---|
| `matchbox.client.*` | `matchlab.*` |
| `matchbox.common.*` | `matchlab.core.*` |
| `matchbox.common.dtos` | `matchlab.core.config` |
| `matchbox.server.*` | *(gone)* |
| `matchbox.client._handler` | *(gone)* |

## The DAG is gone

Previously you created a `DAG`, registered steps on it, and ran it. Now each step holds
its own inputs, so the step you're holding is the pipeline.

=== "Matchbox"

    ```python
    dag = DAG("companies")
    crn = dag.source(location=warehouse, name="crn", ...)
    d = crn.query().deduper(name="d_crn", model_class=NaiveDeduper, model_settings={...})
    r = d.resolver(name="r_crn", resolver_class=Components)
    dag.run_and_sync()
    lookup = dag.get_matches(resolver="r_crn").as_lookup()
    ```

=== "matchlab"

    ```python
    crn = Source(location=warehouse, name="crn", ...)
    entities = crn.dedupe(model_class=NaiveDeduper, model_settings={...}).resolve()
    lookup = entities.collect().get_matches().as_lookup()
    ```

Names are now optional — they're derived from the operation unless you pass one.

## Sources declare less

A source is now its query plus a key. There is no `index_fields` and no `SourceField`.

=== "Matchbox"

    ```python
    Source(
        location=warehouse,
        name="crn",
        extract_transform="select pk, company, town from companies",
        key_field="pk",
        index_fields=["company", "town"],
    )
    ```

=== "matchlab"

    ```python
    Source(
        location=warehouse,
        name="crn",
        extract_transform="select pk, company, town from companies",
        key_field="pk",
    )
    ```

**This changes behaviour, not just signatures.** Identity used to be the indexed fields;
it's now every column the extract returns except the key. If your `index_fields` listed
everything you selected, nothing changes. If it listed *fewer* columns than you selected,
records that used to collapse into one will now stay separate — remove those columns from
the `select` to get the old grouping back.

The upside is that the two can no longer disagree. A column outside the old index could
change in the warehouse without moving the source's fingerprint, so the source cache-hit,
never re-stored, and downstream views kept reading the stale value.

Field types went the same way. They fed one thing — the dtype each column was read as —
and hashing casts every value to text anyway, so the pin only mattered when a driver
changed a column's *kind*. Say it in the SQL instead:

```python
extract_transform = "select pk, cast(crn as text) as crn from companies"
```

Keys are now cast to string on read rather than validated, so an integer primary key
works without a `cast`.

## Renamed operations

Nouns became verbs, and each one is a step in the plan:

| Matchbox | matchlab |
|---|---|
| `source.query(...)` | `source.clean(...)` |
| `query.deduper(...)` | `.dedupe(...)` |
| `query.linker(other, ...)` | `.link(other, ...)` |
| `model.resolver(...)` | `.resolve(...)` |
| `dag.run_and_sync()` | `step.collect()` |
| `dag.get_matches(resolver=...)` | `resolver.get_matches()` |
| `dag.lookup_key(...)` | `resolver.lookup_key(...)` |

## Removed without replacement

**Everything server-side.** Collections, runs, permissions, groups, users, uploads,
`load_default()` / `load_pending()` / `set_default()`, and the whole HTTP layer.

**The CLI.** Every command began by loading a saved DAG by name from the server. Saving
and loading plans doesn't exist yet, so the CLI will return as a feature built on that,
not as a port. The evaluation *library* API is unaffected.

**`low_memory`, `cache_leaf_ids`, `clear_data`.** All inter-step data now flows through
the adapter, so there is nothing held in memory to drop.

**Authentication and client settings.** No `api_root`, no JWT, no `MB__CLIENT__*`
environment variables. Delete them.

**`SourceField`, `Location.infer_types`, and step `description`s.** The first two went
with `index_fields` above. `description` annotated steps for other people to read on the
server; with nothing to serialise it to and nothing to display it in, it was a field you
could set and never observe.

## Exceptions

The exception hierarchy shrank from 40 classes to 8, and dropped the `Matchbox` prefix
on everything but the base:

| Matchbox | matchlab |
|---|---|
| `MatchboxException` | `MatchlabError` |
| `MatchboxStepNotFoundError` | `StepNotFound` |
| `MatchboxArrowSchemaMismatch` | `SchemaMismatch` |
| `MatchboxSourceExtractTransformError` | `ExtractTransformError` |
| `MatchboxSourceClientError` | `SourceClientError` |
| `MatchboxSourceTableError` | `SourceTableError` |
| `MatchboxNameError` | `NameValidationError` |
| `MatchboxRuntimeError` | `DataTypeError` |

Everything else was an HTTP status carrier and went with the server.

## Behaviour that changed

**Nothing runs until `collect()`.** Building a plan is free; only collection does work.

**Results are cached by content.** Re-collecting an unchanged plan does nothing, and
adding a step to a collected plan runs only that step. Rebuilding the same plan in a new
process is a cache hit if the warehouse data is unchanged.

**To refresh a source, construct a new one.** A `Source` object memoises the data it
read. `Source(...)` again re-reads the warehouse and invalidates everything downstream.

**Resolution is materialised, not queried.** A resolver writes a complete
`(root, leaf, key, source)` table when it collects. Queries are reads against that
table, which is why `lookup_key` and `get_matches` are now fast and offline.

**Storage is local and reclaimable.** Artifacts live in a DuckDB store in your user
cache directory by default. Call `matchlab.gc()` to drop artifacts belonging to plans
you no longer hold.

## What didn't change

Matching methodologies (`NaiveDeduper`, `DeterministicLinker`, `SplinkLinker`,
`WeightedDeterministicLinker`), the connected-components resolver, `ResolverMatches`
and its `as_lookup` / `as_dump` / `view_cluster` / `merge` helpers, `Location`
configuration, and the evaluation metrics all behave as before.
