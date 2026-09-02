# Build a plan

A plan is a tree of steps. Each step holds a reference to its inputs, so the node you are holding *is* the pipeline.

Nothing runs until you call `collect()`.

## Sources

A source is the start of any plan. It defines how rows are read, plus the column that keys them.

```python
import matchlab as mb

warehouse = create_engine("postgresql://...")

crn = mb.read_database(
    "crn",
    sql="select pk, company, town from companies",
    client=warehouse,
    key_field="pk",
)
```

Several sources can share one engine. Pass the same object to each.

!!! warning "Your query is run as written"

    matchlab does not parse or sanitise the SQL you give `read_database`. It goes to your client in whatever dialect that client speaks, and a malformed one fails when the source is read, with the database's own error message.

Reading a dataframe you've already shaped needs no query at all:

```python
dh = mb.read_dataframe("dh", df=my_polars_df)
```

The first argument (`name`) prefixes every column this source contributes. For `"crn"`, `company` becomes `crn_company`. A name must start with a letter or underscore, then hold only letters, digits and underscores.

`key_field` is the identifier you'll get back in results. It needs to match an existing column in your source data. It's read as a string whatever type it has in the data. It defaults to `"id"`.

**The identity of each row is defined by all values returned by the source.** Two rows are considered to be the same entity when every column comes back identical for both. Above, a company appearing twice with the same name and town is one entity. Change the town, and it's two. With `read_dataframe` the same rule applies to whatever columns the frame carries. This means:

* **A column you don't want to affect identity is a column that shouldn't come back.** Pull a `last_updated` timestamp through and every row becomes distinct.
* **A type you want pinned is a `cast`** in the SQL, or the frame.
* **Changing the data behind any returned column invalidates the source**, and everything downstream of it.

You can still return a column purely to look at. `view_entity` and the evaluation samplers show every column the source returned, reading it back from the copy cached at collect time, but selecting it still makes it count.

## Verbs

Steps chain. Each verb returns a new lazy step:

| Verb | Produces | Meaning |
|---|---|---|
| `.select(...)` | `Transform` | Keep only the named columns |
| `.clean(...)` | `Transform` | Derive columns with SQL, keeping the rest |
| `.group(...)` | `Transform` | Collapse each `id` to one row |
| `.transform(...)` | `Transform` | Apply any transformer object (the general form) |
| `.dedupe(...)` | `Model` | Candidate matches *within* one step |
| `.link(other, ...)` | `Model` | Candidate matches *between* two steps |
| `.resolve(...)` | `Resolver` | Collapse candidate edges into clusters |
| `.collect()` | (same step) | Run everything the step depends on |

**A source is already matchable.** You don't have to reshape it first. A model reads a source directly:

```python
crn.dedupe(model_class=mb.NaiveDeduper, model_settings={...})
crn.link(dh, model_class=mb.DeterministicLinker, model_settings={...})
```

Both sides of a link are covered. Reach for the reshaping verbs (`select`, `clean`, `group`) only when you want to change what a model sees. They each return a new step, so they chain, and each does one job.

## Reshaping records

A **record step** is a step that produces data for a model to match over. It is not itself a table, but reading one gives a table with an `id` column. On a source or reshaped record step, that `id` is the content-derived identity of the row, not the source key field, which may have another name. On a resolver, `id` is the entity. A `Source`, a resolver, and the output of a reshape step are all record steps. That's why `RecordStep` is what you see in a signature or a hover tooltip wherever one of these is expected. `select`, `clean` and `group` each reshape a record step into a new record step. `transform` is the general verb they translate to. `source.clean(...)` is shorthand for `source.transform(Clean(...))`.

Not every registered transformer has a dedicated verb. The [API reference](../api/plan/steps/transformers) lists all of them, including ones you reach only through `transform(...)`, such as `Explode`.

**`select` keeps only the columns you name**, plus `id`:

```python
crn.select("crn_company", "crn_town")
```

`source.f("field")` gives you the source-qualified column name (`crn_company`), which is how fields are named on a record step.

**`clean` derives columns with DuckDB SQL, keeping the rest**:

```python
cleaned = crn.clean({"name": f"lower({crn.f('company')})"})
```

`name` is added. `crn_company`, `crn_town` and `id` all pass through untouched. Dropping unrelated columns is `select`'s job. Chain them when you want both:

```python
crn.clean({"name": f"lower({crn.f('company')})"}).select("name")
```

### What `id` is, and when to group

Read a source directly and `id` is the record, one row each. **A resolver is itself a
record step**, reshaped with the same verbs. Its `id` is the entity, so several records can
share one:

```python
deduped.clean({"name": "crn_company"}).data()

# id | name
# E1 | acme     <- from a1
# E1 | acme     <- from a2
# E2 | beta
```

That's often what you want, more evidence per entity. When it isn't, **`group` collapses each `id` to one row**. Every expression is an aggregate, so you say how each column combines:

```python
deduped.group(
    {
        "name": "any_value(crn_company)",  # they agree — that's why they grouped
        "towns": "list(distinct crn_town)",  # they differ — keep both
    }
).data()

# id | name | towns
# E1 | acme | ["london", "leeds"]
# E2 | beta | ["hull"]
```

Any DuckDB aggregate works, `list` and `string_agg` included. A non-aggregate gets you DuckDB's own error naming the column. There's no sensible default for how a column collapses, so `group` needs at least one aggregate.

Grouping matters most for a resolver that spans **several** sources. Its rows are concatenated diagonally, so each carries one source's columns and nulls for the rest:

```python
resolver.clean({"c": "crn_company", "d": "dh_company"}).data()

# c    | d
# acme | null      <- from crn
# acme | null      <- from crn
# null | acme      <- from dh
```

A comparison on `l.d` is null on every crn row, so the entity can't be matched on its combined evidence. Grouping puts it on one populated row. `any_value` skips nulls:

```python
resolver.group(
    {
        "company": "any_value(crn_company)",
        "towns": "list(distinct coalesce(crn_town, dh_town))",
    }
).data()

# company | towns
# acme    | ["london", "leeds", "bristol"]
```

Grouping changes what the *model* sees. It never changes the resolver output. Record identity travels separately, so a resolver below a grouped step still carries every record forward.

## Deduplicating and linking

```python
deduped = cleaned.dedupe(
    model_class=mb.NaiveDeduper,
    model_settings={"unique_fields": ["name"]},
)

linked = crn.link(
    dh,
    model_class=mb.DeterministicLinker,
    model_settings={"comparisons": f"l.{crn.f('company')} = r.{dh.f('company')}"},
)
```

Models produce scored *edges*, not clusters. An edge is a scored pair of IDs. The score reflects the model's confidence that those two rows are a match. Turning edges into entities is the resolver's job.

## Resolving

```python
entities = deduped.resolve()
```

`resolve()` defaults to connected components. Pass `resolver_class` and `resolver_settings` for something else.

A resolver takes several models, so you can resolve multiple methodologies together. Give each a score threshold to trust a strict one further than a loose one:

```python
entities = crn_dedupe.resolve(
    dh_dedupe,
    crn_dh_link,
    resolver_settings={"thresholds": {crn_dh_link: 0.9}},
)
```

Thresholds take the model itself, not its name. Any model with no threshold will contribute every edge.

### Layering

A resolver is a record step. To match **on top of** an earlier resolver output, match on the resolver directly with the same verbs. Now `id` means entity, not record:

```python
deduped_crn = crn.dedupe(...).resolve()

entities = deduped_crn.link(  # crn's records, as resolved by the dedupe
    dh, model_class=..., model_settings=...
).resolve()
```

The link now sees crn's deduplicated clusters rather than its raw rows. Records the link never matches keep their upstream grouping. A resolver always carries its inputs' resolver outputs forward, so nothing silently reverts to singletons.

## Collecting

```python
entities.collect()
```

`collect()` walks the plan upstream-first and runs only what isn't already stored.

### Store cache

Steps are content-addressed by their configuration and their inputs' fingerprints, so:

* re-collecting an unchanged plan does no work
* adding a step to a collected plan runs only the new step
* rebuilding the same plan in a new process is a cache hit, provided the source data hasn't changed

Sources are the exception. They hash the data they read, which is how a plan notices that a warehouse table or in-memory dataframe has changed. Constructing a *fresh* `Source` re-reads the data. An existing `Source` object remembers it. If you mutate a dataframe behind `read_dataframe(...)`, redefine the source before collecting again. **Re-collecting the existing source keeps using the memoised read.**

**How the rows were fetched *is* part of the plan.** When reading from a database, the query is hashed alongside the rows, so editing it re-runs the source and everything below, even if the rows come back identical. The client is not hashed. Point the same query at a second engine holding the same rows and you get the same source.

Because the cache key is configuration-derived, it is also conservative. Editing a cleaning expression in a way that doesn't change the data still re-runs everything below it.

### Watching the plan run

In a notebook or terminal, `collect()` draws the plan as a tree and redraws it in place as each step settles:

```
○ [6] resolver(Components)
    ├── ◐ [5] model(NaiveDeduper) running 2.6s
    │   └── ● [3] transform(Clean) 0.4s
    │       └── ● [2] source 'crn' 0.3s
    └── ● [4] model(DeterministicLinker) 0.5s
        ├── ● [3] transform(Clean) 0.4s ↑
        └── ● [1] transform(Clean) 0.2s
            └── ● [0] source 'dh' 0.2s
○ waiting   ◐ running   ● ran
```

This is called *interactive mode*. An interactive plan taller than your screen is "windowed": the frame shows the rows around the running step and says how many are hidden either side, following the run down the tree.

The number in brackets is the step's **position**. It's the same number everywhere. `[5]` here is `[step 5]` in the log and `steps[5]` in a [document](../api/plan/steps). Except for sources, steps have no names, so that cross-reference is how you know which node a line is about.

Next to it is what the step *is*. A model, a resolver and a transform name the class implementing them in parentheses. A source names itself in quotes.

A step feeding two parents is still one step. It's drawn in full where you first meet it, and marked `↑` after. `[3]` above feeds both models but runs once, computed once and read back by each of them. Its inputs are listed only under its first appearance.

`cached` tells you your edit didn't invalidate that step, so nothing was recomputed.

The summary at the end says what the store now costs, and what this run added to it. Here, the store is the DuckDB cache that holds artifacts, as well as the Python object that talks to that store. A store keeps everything you collect into it, so editing a cleaning expression and re-collecting leaves the old artifacts behind. The `(+3.0 MB)` is what tells you which edit did that, while it's still a few megabytes rather than a full disk. A fully cached re-run reads `(+0 B)`. See [Reclaiming storage](#reclaiming-storage).

!!! info "Interactive trees and logs"
    Your own logging handlers work alongside the interactive tree with nothing further to set up. `basicConfig` binds whatever `sys.stderr` was at the time. A running collection borrows handlers pointed at its terminal and routes them through its console until it's finished. Records appear above the tree as they arrive. matchlab's own fallback handler keeps the same split: logs use stderr when stdout is reserved for some other payload.

### Plan logs

When interactive mode is off, the same tree is logged **once**, up front, with each step reporting beneath it:

```
INFO  Collecting 7 steps:
○ [6] resolver(Components)
    ├── ○ [5] model(NaiveDeduper)
    │   └── ○ [3] transform(Clean)
    │       └── ○ [2] source 'crn'
    └── ○ [4] model(DeterministicLinker)
        ├── ○ [3] transform(Clean) ↑
        └── ○ [1] transform(Clean)
            └── ○ [0] source 'dh'
INFO  [step 0] Reading from the warehouse
INFO  [step 0] Ran in 0.160s
INFO  [step 1] Ran in 0.198s
INFO  [step 2] Reading from the warehouse
INFO  [step 2] Ran in 0.255s
INFO  [step 3] Ran in 0.390s
INFO  [step 4] Round 1: Found 2 matches
INFO  [step 4] Ran in 0.515s
INFO  [step 5] Ran in 0.284s
INFO  [step 6] Ran in 0.104s
INFO  Collected 7 steps (7 ran, 0 cached) in 1.402s. Store 3.0 MB (+3.0 MB), 7 artifacts
```

If you don't configure logging, matchlab will automatically add a handler while a collection runs. The output above is what a bare script or a fresh notebook gives you.

If you configure logging, matchlab prints nothing of its own. The records go wherever you send everything else, at the level you chose:

```python
import logging

logging.basicConfig(level=logging.INFO)  # or DEBUG, to see extra logs
```

To turn the reporting off, silence `matchlab` as you would any library:

```python
logging.getLogger("matchlab").addHandler(logging.NullHandler())
```

## Changing a plan

You can print the current version of a plan from a step:

```python
entities.draw()  # the plan, as a tree
entities.lineage()  # every step, inputs first
```

Both look *upstream* only. A source can't reach the resolver built on top of it, because a step knows its inputs and nothing else. `crn.lineage()` is just `[crn]`, however much is built above it.

**A step is settled once it's built.** To change one, rebuild the plan:

```python
def build(unique_fields):
    return (
        crn.clean({"name": "lower(crn_company)"})
        .dedupe(
            model_class=mb.NaiveDeduper, model_settings={"unique_fields": unique_fields}
        )
        .resolve()
    )


entities = build(["name"]).collect()
entities = build(["name", "crn_town"]).collect()  # only the model and resolver re-run
```

The source and the clean above the edit address the same artifacts as before, so they're cache hits. Only the changed step and what's below it run.

Editing in place isn't offered:

```python
model.model_settings = {"unique_fields": ["crn_town"]}
# AttributeError: Model.model_settings is read-only. A step is settled once built,
# so rebuild the plan to change it.
```

Note that if you recollect, the old artifacts stay in the store. See [Reclaiming storage](#reclaiming-storage).

### Step variables

There's no lookup-by-name. To hold on to a step, hold on to the variable.

```python
cleaned = crn.clean({"name": f"lower({crn.f('company')})"})
entities = cleaned.dedupe(...).resolve().collect()

cleaned.data()  # still yours to inspect
```

That goes for settings too. A resolver's per-model thresholds take the model object itself, not its name.

### Collecting again


To collect somewhere other than the default store, pass a new store object to `collect()`:

```python
entities.collect(store=mb.DuckDBStore("./run.duckdb"))
```

## Publishing

To find a resolver output later, **publish** it under a label. That's an operation on the collected result:

```python
entities = crn_dedupe.resolve(dh_dedupe).collect().publish("entities")
```

A label belongs to the store, and points at whichever resolver output you last aimed it at. Republishing the same label for the same resolver output doesn't do anything. Aiming it at a different one needs `overwrite=True`. A plan you never publish still runs, it's just unlabelled.


## Reclaiming storage

**A store keeps everything you collect into it, until you delete the file.** matchlab never removes an artifact on its own initiative.

A step that refreshes stores a new artifact every run rather than replacing the last, so a plan holding one grows the store steadily. This has a cost, so every collect reports it. That's the `Store 3.0 MB (+3.0 MB), 7 artifacts` clause above. You can also ask directly:

```python
stats = mb.default_store().stats()
print(stats.location)  # where the default store actually is
print(stats.bytes)  # what it costs, in bytes
print(stats.artifacts)  # {'source': 8, 'transform': 40, 'model': 32, 'resolver': 24}
print(stats.describe())  # 'Store 1.2 GB, 104 artifacts'
```

Watch the artifact count rather than the size to see this happen. Edit a cleaning expression, re-collect, and the count grows while the plan stays the same size. The old artifacts are still there. For this reason, you may want to prune the store as you go.


### Pruning

`prune()` keeps what you name and deletes the rest:

```python
entities = build_plan().collect()

result = mb.default_store().prune(keep=entities.fingerprints())
print(result.describe())
# 'Removed 80 artifacts, kept 24, reclaimed 416.2 MB'
```

`plan.fingerprints()` names every artifact a plan is made of, its own and its inputs'. Which artifacts those are is the plan's business, not the store's. `prune()` takes those fingerprints directly.

**Published labels are kept whether or not you list their fingerprints**. A published resolver output also keeps the source extracts it needs, so you can still read it back later without the plan.

Nothing is inferred about what you are still using. matchlab does not watch which objects your program is holding. A store outlives the process that wrote it. You say what to keep.

Pruning rewrites the store, which reopens its connection. Any session setting applied through `store.conn` (see [Keeping memory bounded](#keeping-memory-bounded)) has to be applied again afterwards.

### Starting from cold

Deleting the file is the way to throw everything away:

```python
from pathlib import Path

Path("./run.duckdb").unlink()  # start again from cold
```

The default store lives in your user cache directory. `mb.default_store().stats().location` will tell you exactly where, and it's safe to delete at any time. You lose cache hits, not results you can't rebuild, provided the warehouse data hasn't moved.

!!! warning "DuckDB files do not shrink"
    Deleting rows or dropping tables inside a DuckDB file does **not** return space to the operating system. DuckDB marks the blocks free and reuses them for later writes, but the file stays the size of its high-water mark. There is no `VACUUM FULL`, and `CHECKPOINT` will not do it either:

    ```
    after write:               149.7 MB
    after DROP + CHECKPOINT:   149.7 MB
    ```

    This is why `prune()` **rewrites** the store rather than deleting inside it. Purging artifacts alone would buy reuse headroom while your disk usage stayed exactly the same. A prune reports what it actually recovered, measured before and after.

### Keeping memory bounded

An in-memory store (`mb.DuckDBStore(":memory:")`) is not limited to RAM. DuckDB spills table data to a temporary directory once it exceeds `memory_limit`. That defaults to about 80% of your machine's memory:

```python
store = mb.DuckDBStore(":memory:")
store.conn.execute("SET memory_limit = '4GB'")
store.conn.execute("SET temp_directory = '/fast/scratch'")
```

That bounds the resident footprint without discarding anything. That's almost always what you want from a cache. Paged out is cheap to read back, deleted has to be recomputed.
