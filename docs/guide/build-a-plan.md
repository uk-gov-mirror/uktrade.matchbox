# Build a plan

A plan is a tree of steps. Each step holds a reference to its inputs, so the node you
are holding *is* the pipeline — there is no separate container object to register
things with.

Nothing runs until you call `collect()`.

## Sources

A source is a leaf: a warehouse query, plus the column that keys it.

```python
from matchlab import Source

crn = Source(
    location=warehouse,
    name="crn",
    extract_transform="select pk, company, town from companies",
    key_field="pk",
)
```

`key_field` is the identifier you'll get back in results. It's read as a string
whatever the warehouse stores it as, so an integer primary key needs no ceremony.

`name` qualifies every column this source contributes — `company` becomes
`crn_company` — and those names end up in cleaning SQL, so it has to work as the start
of a column name: a letter or underscore, then letters, digits and underscores. A
hyphen or a dot would parse as arithmetic or as a table reference, so matchlab rejects
them when you build the source rather than letting it fail three steps later. SQL
keywords are fine, since the name is only ever a prefix.

**The `select` is the whole declaration.** Every other column it returns is part of the
record, and so part of that record's identity: two rows are the same record exactly
when the extract returns identical values for both. Above, a company appearing twice
with the same name and town is one record; change the town and it's two.

There's no separate list of fields to index. That means:

* **A column you don't want to affect identity is a column you shouldn't select.** Pull
  a `last_updated` timestamp through and every row becomes distinct.
* **A type you want pinned is a `cast` in the SQL.** `select cast(crn as text)` reads
  more directly than a parallel type system, and it can't drift from the query.
* **Changing the warehouse data behind any selected column invalidates the source**, and
  everything downstream of it.

You can still fetch a column purely to look at — `view_cluster` and the evaluation
samplers re-read the warehouse through the same `extract_transform` — but be aware that
selecting it makes it count.

## Verbs

Steps chain. Each verb returns a new lazy step:

| Verb | Produces | Meaning |
|---|---|---|
| `.view(...)` | `View` | The records a model matches over, optionally reshaped |
| `.dedupe(...)` | `Model` | Candidate matches *within* one view |
| `.link(other, ...)` | `Model` | Candidate matches *between* two views |
| `.resolve(...)` | `Resolver` | Collapse candidate edges into clusters |
| `.collect()` | (same step) | Run everything the step depends on |

**You don't have to call `.view()`.** Every model needs one, but matchlab inserts it
for you when you go straight from a source:

```python
crn.dedupe(model_class=NaiveDeduper, model_settings={...})
crn.link(dh, model_class=DeterministicLinker, model_settings={...})
```

Both sides of a link are covered — passing a `Source` where a view is expected views
it. Reach for `.view()` when you want to do one of the three things only a view can do:
clean columns, `group`, or read through a resolver.

### Views

A view says **which records a model matches over, and what shape they're in**. Both
halves matter, and the first is the one that's easy to miss.

```python
cleaned = crn.view(
    {
        "name": f"lower({crn.f('company')})",
        "town": crn.f("town"),
    }
)
```

Only the columns you name survive, plus `id` — the grouping the model matches on.
`source.f("field")` gives you the source-qualified column name (`crn_company`), which
is how fields are named once a view is built.

!!! note
    `view()` means "no cleaning" and passes every column through. `view(cleaning={})`
    is a real projection that selects nothing, leaving only `id`. They are deliberately
    different — "I didn't ask for a projection" and "I asked for an empty one".

#### What `id` is, and when to group

Read a source directly and `id` is the record — one row each. Read it *through a
resolver* and `id` is the resolver's entity, so several records share one:

```python
deduped.view(crn, cleaning={"name": "crn_company"}).data()

# id | name
# E1 | acme     <- from a1
# E1 | acme     <- from a2
# E2 | beta
```

That's often what you want — more evidence per entity. When it isn't, `group=True`
collapses each `id` to one row, and every expression becomes an aggregate so you say
how each column combines:

```python
deduped.view(
    crn,
    cleaning={
        "name": "any_value(crn_company)",  # they agree — that's why they grouped
        "towns": "list(distinct crn_town)",  # they differ — keep both
    },
    group=True,
).data()

# id | name | towns
# E1 | acme | ["london", "leeds"]
# E2 | beta | ["hull"]
```

Any DuckDB aggregate works, `list` and `string_agg` included. A non-aggregate gets you
DuckDB's own error naming the column. `group=True` needs cleaning expressions — there's
no sensible default for how a column collapses.

Grouping matters most when a view reads **several** sources through a resolver. Those
are concatenated diagonally, so each row carries one source's columns and nulls for the
rest:

```python
resolver.view(crn, dh, cleaning={"c": "crn_company", "d": "dh_company"}).data()

# c    | d
# acme | null      <- from crn
# acme | null      <- from crn
# null | acme      <- from dh
```

A comparison on `l.d` is null on every crn row, so the entity can't be matched on its
combined evidence. Grouping puts it on one populated row — `any_value` skips nulls:

```python
resolver.view(
    crn,
    dh,
    cleaning={
        "company": "any_value(crn_company)",
        "towns": "list(distinct coalesce(crn_town, dh_town))",
    },
    group=True,
).data()

# company | towns
# acme    | ["london", "leeds", "bristol"]
```

Grouping changes what the *model* sees, never the resolution: record identity travels
separately, so a resolver below a grouped view still carries every record forward.

### Deduplicating and linking

```python
from matchlab.models.dedupers import NaiveDeduper
from matchlab.models.linkers import DeterministicLinker

deduped = cleaned.dedupe(
    model_class=NaiveDeduper,
    model_settings={"unique_fields": ["name"]},
)

linked = crn.view().link(
    dh,
    model_class=DeterministicLinker,
    model_settings={"comparisons": f"l.{crn.f('company')} = r.{dh.f('company')}"},
)
```

Models produce scored *edges*, not clusters. Turning edges into entities is the
resolver's job.

### Resolving

```python
entities = deduped.resolve()
```

`resolve()` defaults to connected components. Pass `resolver_class` and
`resolver_settings` for something else.

A resolver takes several models, so you can resolve multiple methodologies together —
and trust a strict one further than a loose one by giving each a score threshold:

```python
entities = crn_dedupe.resolve(
    dh_dedupe,
    crn_dh_link,
    resolver_settings={"thresholds": {crn_dh_link: 0.9}},
)
```

Thresholds take the model itself, not its name — you're already holding it. Any model with no threshold will contribute every edge.

## Layering

To match *on top of* an earlier resolution, read a source through it:

```python
deduped_crn = crn.dedupe(...).resolve()

entities = (
    deduped_crn.view(crn)  # crn, as resolved by the dedupe
    .link(dh, model_class=..., model_settings=...)
    .resolve()
)
```

The link now sees crn's deduplicated clusters rather than its raw rows. Records the
link never matches keep their upstream grouping — a resolver always carries its inputs'
resolutions forward, so nothing silently reverts to singletons.

## Collecting

```python
entities.collect()
```

`collect()` walks the plan upstream-first and runs only what isn't already stored.
Steps are content-addressed by their configuration and their inputs' fingerprints, so:

* re-collecting an unchanged plan does no work;
* adding a step to a collected plan runs only the new step;
* rebuilding the same plan in a new process is a cache hit, provided the warehouse data
  hasn't changed.

Sources are the exception — they hash the data they read, which is how a plan notices
the warehouse moved. Constructing a *fresh* `Source` re-reads it; an existing `Source`
object remembers.

!!! warning "Seed anything non-deterministic"
    A step's cache key comes from its configuration, not from its output. If a model
    can produce different results from the same settings, the first result is cached
    and reused. In practice this means passing a `seed` to Splink training functions
    that sample — otherwise re-running gives you the cache, not a second opinion.

Because the key is configuration-derived, it is also conservative: editing a cleaning
expression in a way that doesn't change the data still re-runs everything below it.

To collect somewhere other than the default store:

```python
entities.collect(adapter=DuckDBAdapter("./run.duckdb"))
```

## Inspecting

```python
entities.draw()  # the plan, as a tree
entities.lineage()  # every step, inputs first
```

Both look *upstream* only. A source can't reach the resolver built on top of it,
because a step knows its inputs and nothing else — `crn.lineage()` is just `[crn]`
however much is built above it.

There's no lookup-by-name: to hold on to a step, hold on to the variable.

```python
cleaned = crn.view(cleaning={"name": f"lower({crn.f('company')})"})
entities = cleaned.dedupe(...).resolve().collect()

cleaned.data()  # still yours to inspect
```

That goes for settings too: a resolver's per-model thresholds take the model itself,
not its name.

Steps have no names at all. To find a resolution later, **publish** it under a label —
an operation on the collected result, not a property of the plan:

```python
entities = crn_dedupe.resolve(dh_dedupe).collect().publish("entities")
```

Republishing the same label for the same resolution is a no-op; aiming it at a
different one needs `overwrite=True`. A plan you never publish still runs — it is just
unlabelled.

A label is not a name. A *name* belongs to a source and is part of its output; a label
belongs to the store, and points at whichever resolution you last aimed it at.

Everything else goes by **position**: the order `collect` runs it in, which is what
logs quote and what `draw()` shows in brackets.

```
[step 2] Ran in 0.041s
```

```python
print(entities.draw())
```
```
○ [4] resolver 'entities'
    ├── ○ [3] model
    │   └── ○ [1] clean
    │       └── ○ [0] source 'crn'
    └── ○ [2] model
        └── ○ [1] clean
            └── ○ [0] source 'crn'
```

Positions are relative to the apex you collected or drew from, so a plan and a
sub-plan of it number differently — but a run and that run's drawing always agree.

## Reclaiming storage

Collected artifacts stay in the adapter. When you drop a plan, its storage becomes
reclaimable:

```python
import matchlab

matchlab.gc()  # drop artifacts no live step references
```

## Next

[Query the result :octicons-arrow-right-16:](./query.md){ .md-button .md-button--primary }
