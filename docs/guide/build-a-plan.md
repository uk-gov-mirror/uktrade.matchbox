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
| `.clean(...)` | `Cleaner` | A queryable view, optionally with cleaning SQL |
| `.dedupe(...)` | `Model` | Candidate matches *within* one view |
| `.link(other, ...)` | `Model` | Candidate matches *between* two views |
| `.resolve(...)` | `Resolver` | Collapse candidate edges into clusters |
| `.collect()` | (same step) | Run everything the step depends on |

### Cleaning

`clean` takes a mapping of output column to SQL expression. Only the columns you name
survive — plus the identifiers, which always pass through.

```python
cleaned = crn.clean(
    {
        "name": f"lower({crn.f('company')})",
        "town": crn.f("town"),
    }
)
```

`source.f("field")` gives you the source-qualified column name (`crn_company`), which
is how fields are named once a view is built.

!!! note
    `clean(None)` means "no cleaning" and passes every column through. `clean({})` is a
    real projection that selects nothing, leaving only the identifiers. They are
    deliberately different.

### Deduplicating and linking

```python
from matchlab.models.dedupers import NaiveDeduper
from matchlab.models.linkers import DeterministicLinker

deduped = cleaned.dedupe(
    model_class=NaiveDeduper,
    model_settings={"unique_fields": ["name"]},
)

linked = crn.clean().link(
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
`resolver_settings` for something else — for example per-model score thresholds.

A resolver takes several models, so you can resolve multiple methodologies together:

```python
entities = crn_dedupe.resolve(dh_dedupe, crn_dh_link, name="entities")
```

## Layering

To match *on top of* an earlier resolution, read a source through it:

```python
deduped_crn = crn.dedupe(...).resolve()

entities = (
    deduped_crn.clean(crn)  # crn, as resolved by the dedupe
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
entities.get_step("crn")  # a step by name, searching only this plan's lineage
entities.lineage()  # every step, inputs first
```

`get_step` looks *upstream* only. A source can't reach the resolver built on top of it,
because a step knows its inputs and nothing else.

## Reclaiming storage

Collected artifacts stay in the adapter. When you drop a plan, its storage becomes
reclaimable:

```python
import matchlab

matchlab.gc()  # drop artifacts no live step references
```

## Next

[Query the result :octicons-arrow-right-16:](./query.md){ .md-button .md-button--primary }
