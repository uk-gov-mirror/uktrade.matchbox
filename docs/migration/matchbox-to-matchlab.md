# Matchbox to matchlab

matchlab is Matchbox with the server removed and the client API rebuilt around a lazy
plan. This is a hard break: a new package name, a new import root, and a different way
of expressing a pipeline.

There is no compatibility shim. `matchbox-db` receives no further releases.

## Why

The server existed mainly because you needed it to run a DAG. In practice the
collaboration features it enabled went unused. Judgements weren't written back, teams
didn't share collections, and nobody built on each other's DAGs. What it did cost was
real: a large codebase, a worse interface, slower pipelines, and no way for analysts
outside government to adopt it at all.

Removing it deleted more than half the code. It also made the pipeline faster, because
the resolver output is now computed once when you collect, rather than re-derived on
every query.

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
| `matchbox.common.dtos` | `matchlab.specs` |
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
    companies = crn.dedupe(model_class=NaiveDeduper, model_settings={...}).resolve()
    lookup = companies.collect().get_lookup()
    ```

## Publishing replaced naming

Steps have no names. In Matchbox every step took one. Now a name is not part of a plan
at all, because it changes nothing about what gets computed. **Publishing a resolver's
output under a label is an operation you perform on the result:**

```python
entities = crn_dedupe.resolve(dh_dedupe).collect().publish("entities")
```

That label is what `matchlab review <label>` and `get_samples(resolver=...)` find. A
plan you never publish still runs perfectly well. It is just unlabelled, addressed by
fingerprint like everything else.

Republishing the same label for the same resolver output is a no-op, so re-running an
unchanged pipeline is safe. Aiming an existing label somewhere new is deliberate:

```python
entities.publish("entities", overwrite=True)
```

**Sources keep their name.** It is a different thing entirely: it prefixes every
column that source contributes (`crn_company`) and tags its rows in a resolver output.
It is part of the output, not a way of finding it.

Two things shared the name `name`, but did very different jobs. That is why one job
became an operation instead. Its result is called a *label*, so the word is now
unambiguous. A name belongs to a source. A label belongs to a store.

### Everything else goes by position

A step is identified by its position in the plan, the order `collect` runs it in.
Trying two settings of a methodology over one view needs no names, and cannot collide.
Before, each setting needed a name you'd never use again:

```python
view = crn.view(cleaning={...})
strict = view.dedupe(NaiveDeduper, {"unique_fields": ["name", "town"]})
loose = view.dedupe(NaiveDeduper, {"unique_fields": ["name"]})
entities = strict.resolve(loose).collect().publish("entities")
```

Logs quote the position:

```
[step 0] Reading from the warehouse
[step 0] Ran in 0.004s
[step 2] Ran in 0.041s
[step 3] Ran in 0.012s
[step 4] Ran in 0.005s
```

`draw()` numbers the same walk the same way:

```
● [4] resolver(Components)
    ├── ● [3] model(NaiveDeduper)
    │   └── ● [1] view
    │       └── ● [0] source 'crn'
    └── ● [2] model(NaiveDeduper)
        └── ● [1] view ↑
```

The shared view appears as `[1]` under both models, because it *is* one node read
twice, not two identical ones. It is drawn in full where you first meet it, and marked
`↑` after that. It runs once, and both models read its stored table.

### Per-model thresholds take the model, not its name

```python
# Matchbox
resolver_settings = {"thresholds": {"d_crn": 0.9}}

# matchlab
resolver_settings = {"thresholds": {d_crn: 0.9}}
```

You already hold the model, so there's nothing to retype and nothing to keep in sync.
A typo now raises `NameError` straight away, where it used to be a runtime failure at
collect time. Thresholds are stored by input position, which is also why models need
no names for this to work.

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

**This changes behaviour, not just signatures.** Identity used to be the indexed fields.
It's now every column the extract returns except the key. If your `index_fields` listed
everything you selected, nothing changes. If it listed *fewer* columns than you selected,
records that used to collapse into one now stay separate. Remove those columns from
the `select` to get the old grouping back.

The upside is that the two can no longer disagree. A column outside the old index could
change in the warehouse without moving the source's fingerprint. The source then
registered a cache hit instead of re-storing, and downstream views kept reading the
stale value.

Field types went the same way. They served one purpose, the dtype each column was read
as. Hashing casts every value to text anyway, so the pin only mattered when a driver
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
| `source.query(...)` | `source.view(...)` |
| `query.deduper(...)` | `.dedupe(...)` |
| `query.linker(other, ...)` | `.link(other, ...)` |
| `model.resolver(...)` | `.resolve(...)` |
| `dag.run_and_sync()` | `step.collect()` |
| `dag.get_matches(resolver=...).as_lookup()` | `resolver.get_lookup()` |
| `dag.lookup_key(...)` | `resolver.lookup_key(...)` |

`query` became `view` rather than `clean`: the node's job is to say which records a
model matches over, and cleaning is an optional clause of that.

`cleaning` is now **keyword-only**, so `source.view(cleaning={...})` reads the same as
`resolver.view(source, cleaning={...})`. The positional slot is taken by the sources
being read.

### `QueryCombineType` is gone

`concat` was the default and is now the only ungrouped behaviour: one row per record,
even when reading through a resolver. The other two are worth explaining, because one
of them didn't do what its name suggests.

**`set_agg`** collapsed each entity to one row, but wrapped *every* column in a list.
That included the column you deduped on, whose values agree by construction anyway.
Comparison-based matchers can't consume a list, which is why it went unused.

**`explode`** looks like it should have produced the cross-product of each entity's
values across sources. That's what you'd want when a view reads several sources and
each row carries one source's columns and nulls for the rest. It didn't. It grouped
every column into a list and then exploded them *in parallel*, and Polars explodes
multiple columns element-wise rather than as a cross product:

```python
{"a": [["x", "y"]], "b": [["p", "q"]]}  # explode("a", "b") gives (x, p), (y, q)
```

Since every list held one entry per row, that round-tripped straight back to the input.
`explode` was therefore `concat` plus a `unique()` and a reordering. It was a broken
feature, not a redundant one.

**Use `group=True`** with aggregate cleaning expressions instead. It does what `explode`
was reaching for, and lets you choose per column. Because DuckDB's `any_value` skips
nulls, it collapses a multi-source view onto one populated row:

```python
resolver.view(
    crn,
    dh,
    cleaning={
        "company": "any_value(crn_company)",
        "towns": "list(distinct coalesce(crn_town, dh_town))",
    },
    group=True,
)
```

One thing is genuinely unavailable: a true cross product, N left rows × M right rows.
`group` gives one row per entity. That is the right unit for a linker, since a cross
product multiplies a single entity's evidence rather than combining it. Nothing offers
a true cross product, if you need one anyway.

## Removed without replacement

**Everything server-side.** Collections, runs, permissions, groups, users, uploads,
`load_default()` / `load_pending()` / `set_default()`, and the whole HTTP layer.

**Most of the CLI.** `health`, `auth`, `collections`, `groups` and `admin` were all
server operations. `matchbox eval` survives as `matchlab review`, but it names a plan
differently: there is no `--collection` to fetch by name, so it takes a
`module:attribute` pointing at a resolver in your own code.

```shell
matchbox eval --collection companies --warehouse postgresql://...  # before
matchlab review pipeline:entities                                  # after
```

`--warehouse` is gone with it: the plan is Python, so it already has its clients.

**`low_memory`, `cache_leaf_ids`, `clear_data`.** All inter-step data now flows through
the adapter, so there is nothing held in memory to drop.

**Authentication and client settings.** No `api_root`, no JWT, no `MB__CLIENT__*`
environment variables. Delete them.

**`SourceField`, `Location.infer_types`, and step `description`s.** The first two went
with `index_fields` above. `description` annotated steps for other people to read on the
server. With nothing to serialise it to, and nothing to display it in, it became a field
you could set but never observe.

**`Location.set_client`.** A location took its client separately because it used to be
half of a server-side row, rebuilt from a config and given a client afterwards. Pass it
to the constructor instead:

```python
# Matchbox
location = RelationalDBLocation(name="warehouse")
location.set_client(engine)

# matchlab
location = RelationalDBLocation(name="warehouse", client=engine)
```

A location now always has a client, and is immutable from the moment it exists, so
`SourceClientError` and every "is the client set?" check went with it.

## Exceptions

The exception hierarchy shrank from 40 classes to 6, and dropped the `Matchbox` prefix
on everything but the base:

| Matchbox | matchlab |
|---|---|
| `MatchboxException` | `MatchlabError` |
| `MatchboxStepNotFoundError` | `StepNotFound` |
| `MatchboxArrowSchemaMismatch` | `SchemaMismatch` |
| `MatchboxSourceExtractTransformError` | `ExtractTransformError` |
| `MatchboxSourceClientError` | *gone* — a `Location` takes its client in `__init__`, so it can never be used without one |
| `MatchboxSourceTableError` | `SourceTableError` |
| `MatchboxNameError` | *gone* — `Source` validates its own name |
| `MatchboxRuntimeError` | `MatchlabError` |

Everything else was an HTTP status carrier and went with the server.

## Behaviour that changed

**Nothing runs until `collect()`.** Building a plan is free. Only collection does work.

**Results are cached by content.** Re-collecting an unchanged plan does nothing, and
adding a step to a collected plan runs only that step. Rebuilding the same plan in a new
process is a cache hit if the warehouse data is unchanged.

**To refresh a source, construct a new one.** A `Source` object memoises the data it
read. `Source(...)` again re-reads the warehouse and invalidates everything downstream.

**A resolver output is materialised, not queried.** A resolver writes a complete
`(root, leaf, key, source)` table when it collects. Queries are reads against that
table, which is why `lookup_key` and `entities` are now fast and offline.

**Storage is local and kept until you say otherwise.** Artifacts live in a DuckDB store
in your user cache directory by default, and every collect reports what it costs.
Nothing is removed on the library's initiative: either delete the store file, or call
`trim(keep=...)` and name what to preserve. A DuckDB file does not shrink when rows are
deleted, so trimming rewrites the store to give the space back. See the guide's
"Reclaiming storage".

## What didn't change

Matching methodologies (`NaiveDeduper`, `DeterministicLinker`, `SplinkLinker`,
`WeightedDeterministicLinker`), the connected-components resolver, `Location`
configuration, and the evaluation metrics all behave as before.

### Reads moved onto the resolver

`ResolverMatches` is gone. Every one of its methods was a projection of the resolver
output the resolver already materialises. They now sit directly on `Resolver`, with no
intermediate object and no second spelling of `root` and `leaf`.

| Matchbox | matchlab |
|---|---|
| `matches.as_dump()` | `resolver.entities()` |
| `matches.as_lookup()` | `resolver.get_lookup()` |
| `matches.as_leaf_sets()` | `resolver.leaf_sets()` |
| `matches.view_cluster(id)` | `resolver.view_entity(root)` |
| `get_matches(source_filter=[...])` | `entities(sources=[...])`, and the same on the rest |
| `ResolverMatches.from_dump(...)` | publish a label, and read the store |
| `matches.merge(other)` | `get_samples`/`review` over several resolvers |

Two behavioural changes worth knowing about:

* **`view_entity` reads the store, not the warehouse.** `view_cluster` re-read the
  source through its `extract_transform`, so it could show different values from the
  ones the reviewer had on screen. It now reads the extract cached at collect time.
  That's the data the matching actually saw, so it needs no connection.
* **Comparing methodologies is an evaluation job.** `merge()` unioned two resolvers'
  components client-side so you could dump them to parquet and review the merged
  clusters. `get_samples`, `review` and `EvalData.precision_recall` now take several
  resolvers directly, and `review(..., seed=...)` replaces passing that file around.
