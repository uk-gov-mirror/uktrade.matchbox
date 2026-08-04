# Build a plan

A plan is a tree of steps. Each step holds a reference to its inputs, so the node you
are holding *is* the pipeline. There is no separate container object to register
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

`name` qualifies every column this source contributes. `company` becomes
`crn_company`. Those names end up in cleaning SQL, so a name has to work as the start
of a column name. That means a letter or underscore, then letters, digits and underscores. A
hyphen or a dot would parse as arithmetic or as a table reference. This means matchlab rejects
an invalid name when you build the source, rather than letting it fail three steps
later. SQL keywords are fine, since the name is only ever a prefix.

**The `select` is the whole declaration.** Every column it returns becomes part of the
record, and so part of that record's identity. Two rows are the same record exactly
when the extract returns identical values for both. Above, a company appearing twice
with the same name and town is one record. Change the town, and it's two.

There's no separate list of fields to index. That means:

* **A column you don't want to affect identity is a column you shouldn't select.** Pull
  a `last_updated` timestamp through and every row becomes distinct.
* **A type you want pinned is a `cast` in the SQL.** `select cast(crn as text)` reads
  more directly than a parallel type system, and it can't drift from the query.
* **Changing the warehouse data behind any selected column invalidates the source**, and
  everything downstream of it.

You can still select a column purely to look at. `view_entity` and the evaluation
samplers show every column the extract returned, reading it back from the copy cached
at collect time, but selecting it still makes it count.

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

Both sides of a link are covered. Passing a `Source` where a view is expected views
it. Reach for `.view()` when you want to do one of the three things only a view can
do: clean columns, `group`, or read through a resolver.

### Views

A view says **which records a model matches over, and what shape they're in**. Both
halves matter, and the first is the one that's easy to miss.

```python
cleaned = crn.view(
    cleaning={
        "name": f"lower({crn.f('company')})",
        "town": crn.f("town"),
    }
)
```

Only the columns you name survive, plus `id`. That's the grouping the model matches on.
`source.f("field")` gives you the source-qualified column name (`crn_company`), which
is how fields are named once a view is built.

!!! note
    `view()` means "no cleaning" and passes every column through. `view(cleaning={})`
    is a real projection that selects nothing, leaving only `id`. They are deliberately
    different — "I didn't ask for a projection" and "I asked for an empty one".

#### What `id` is, and when to group

Read a source directly and `id` is the record, one row each. Read it *through a
resolver* and `id` is the resolver's entity, so several records share one:

```python
deduped.view(crn, cleaning={"name": "crn_company"}).data()

# id | name
# E1 | acme     <- from a1
# E1 | acme     <- from a2
# E2 | beta
```

That's often what you want — more evidence per entity. When it isn't, `group=True`
collapses each `id` to one row. Every expression then becomes an aggregate, so you say
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
DuckDB's own error naming the column. `group=True` needs cleaning expressions. There's
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
combined evidence. Grouping puts it on one populated row. `any_value` skips nulls:

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

Grouping changes what the *model* sees. It never changes the resolution. Record
identity travels separately, so a resolver below a grouped view still carries every
record forward.

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

A resolver takes several models, so you can resolve multiple methodologies together.
Give each a score threshold to trust a strict one further than a loose one:

```python
entities = crn_dedupe.resolve(
    dh_dedupe,
    crn_dh_link,
    resolver_settings={"thresholds": {crn_dh_link: 0.9}},
)
```

Thresholds take the model itself, not its name. You're already holding it. Any model
with no threshold will contribute every edge.

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
link never matches keep their upstream grouping. A resolver always carries its inputs'
resolutions forward, so nothing silently reverts to singletons.

## Collecting

```python
entities.collect()
```

`collect()` walks the plan upstream-first and runs only what isn't already stored.
Steps are content-addressed by their configuration and their inputs' fingerprints, so:

* re-collecting an unchanged plan does no work
* adding a step to a collected plan runs only the new step
* rebuilding the same plan in a new process is a cache hit, provided the warehouse data
  hasn't changed

Sources are the exception. They hash the data they read, which is how a plan notices
the warehouse has moved. Constructing a *fresh* `Source` re-reads the data. An
existing `Source` object remembers it.

!!! warning "Seed anything non-deterministic"
    A step's cache key comes from its configuration, not from its output. If a model
    can produce different results from the same settings, the first result is cached
    and reused. In practice, this means passing a `seed` to Splink training functions
    that sample. Otherwise, re-running gives you the cache, not a second opinion.

Because the key is configuration-derived, it is also conservative. Editing a cleaning
expression in a way that doesn't change the data still re-runs everything below it.

To collect somewhere other than the default store:

```python
entities.collect(adapter=DuckDBAdapter("./run.duckdb"))
```

### Watching it run

At a terminal, `collect()` draws the plan as a tree and redraws it in place as each
step settles. That's one frame, not one tree per step:

```
○ [6] resolver(Components)
    ├── ◐ [5] model(NaiveDeduper) running 2.6s
    │   └── ● [3] view 0.4s
    │       └── ● [2] source 'crn' 0.3s
    └── ● [4] model(DeterministicLinker) 0.5s
        ├── ● [3] view 0.4s ↑
        └── ● [1] view 0.2s
            └── ● [0] source 'dh' 0.2s
○ waiting   ◐ running   ● ran
```

The number in brackets is the step's **position**. It's the same number everywhere.
`[5]` here is `[step 5]` in the log and `steps[5]` in a
[document](../api/steps.md). Steps have no names, so that cross-reference is how
you know which node a line is about. That's why every mode puts the tree somewhere.

Next to it is what the step *is*. A model and a resolver name the class implementing
them in parentheses, so `[5]` reads as the naive dedupe rather than as another
anonymous `model` line. A source names itself in quotes, since a source is the one
step with a name. A view is just a view.

The legend lists only what's on screen. A node feeding two branches is still one
node. It's drawn in full where you first meet it, and marked `↑` after. `[3]` above
feeds both models but runs once, computed once and read back by each of them. Its
inputs are listed only under its first appearance, not repeated. On plans with a
shared base, that's the difference between a readable tree and a few hundred lines.

`cached` is the one to watch. It tells you your edit didn't invalidate that step, so
nothing was recomputed.

Where nothing is drawn (a scheduler, CI, a redirected stream), the same tree is logged
**once**, up front, with each step reporting beneath it:

```
INFO  Collecting 7 steps:
○ [6] resolver(Components)
    ├── ○ [5] model(NaiveDeduper)
    │   └── ○ [3] view
    │       └── ○ [2] source 'crn'
    └── ○ [4] model(DeterministicLinker)
        ├── ○ [3] view ↑
        └── ○ [1] view
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

The summary also says what the store now costs, and what this run added to it. A store
keeps everything you collect into it, so editing a cleaning expression and
re-collecting leaves the old artifacts behind. The `(+3.0 MB)` is what tells you which
edit did that, while it's still a few megabytes rather than a full disk. A fully
cached re-run reads `(+0 B)`. See [Reclaiming storage](#reclaiming-storage).

Work done is logged at `INFO`. Skipping a cached step is logged at `DEBUG`, and the
closing summary totals those so an `INFO` reader still sees what the run avoided.
Anything a step logs while it runs is prefixed the same way, so a linker reporting its
rounds lands under the position it belongs to. Like any library logger, it's silent
until you configure logging:

```python
import logging

logging.basicConfig(level=logging.INFO)
```

The plan is put in **one** place, never two. A drawn tree is already the key those
`[step N]` lines need. It's on screen throughout, and left there in full when the run
ends, so it isn't logged as well. The per-step records are the same either way.

That choice is `collect(interactive=...)`, named for the assumption it makes rather
than the widget it produces. Drawing means *someone is watching*, so the plan can be a
thing on screen that the session throws away. `interactive=False` puts the tree in the
log instead. That's what a run whose output outlives the session wants. The default,
`interactive=None`, reads a terminal or a notebook as a yes, and anything else as a no.

A plan taller than your window is windowed rather than dropped. The frame shows the
rows around the running step and says how many are hidden either side, following the
run down the tree. The last frame is the whole thing.

```
⋮ 4 more above · 27 more below
    │   │   └── ○ [12] resolver(Components)
    │   │       ├── ○ [11] model(NaiveDeduper)
    │   │       │   └── ○ [10] view
    │   │       │       └── ○ [9] source 'crn' ↑
    │   │       ├── ◐ [8] model(NaiveDeduper) running 0.4s
    │   │       │   └── ◍ [7] view cached
    │   │       │       └── ◍ [6] source 'dh' cached
    │   │       ├── ◍ [5] model(NaiveDeduper) cached
○ waiting   ◐ running   ◍ cached
```

That works alongside the live tree with nothing further to set up. `basicConfig` binds
whatever `sys.stderr` was at the time. Left alone, that would write over the frame
being redrawn, so a running collection borrows handlers pointed at its terminal and
routes them through its console until it's finished. Records appear above the tree as
they arrive.

## Inspecting

```python
entities.draw()  # the plan, as a tree
entities.lineage()  # every step, inputs first
```

Both look *upstream* only. A source can't reach the resolver built on top of it,
because a step knows its inputs and nothing else. `crn.lineage()` is just `[crn]`,
however much is built above it.

There's no lookup-by-name. To hold on to a step, hold on to the variable.

```python
cleaned = crn.view(cleaning={"name": f"lower({crn.f('company')})"})
entities = cleaned.dedupe(...).resolve().collect()

cleaned.data()  # still yours to inspect
```

That goes for settings too. A resolver's per-model thresholds take the model itself,
not its name.

Steps have no names at all. To find a resolution later, **publish** it under a
label. That's an operation on the collected result, not a property of the plan:

```python
entities = crn_dedupe.resolve(dh_dedupe).collect().publish("entities")
```

Republishing the same label for the same resolution is a no-op. Aiming it at a
different one needs `overwrite=True`. A plan you never publish still runs. It's just
unlabelled.

A label is not a name. A *name* belongs to a source and is part of its output. A
label belongs to the store, and points at whichever resolution you last aimed it at.

Everything else goes by **position**. That's the order `collect` runs it in, which is what
logs quote and what `draw()` shows in brackets.

```
[step 2] Ran in 0.041s
```

```python
print(entities.draw())
```
```
● [6] resolver(Components)
    ├── ● [5] model(NaiveDeduper)
    │   └── ● [4] view
    │       └── ● [3] source 'crn'
    └── ● [2] model(NaiveDeduper)
        └── ● [1] view
            └── ● [0] source 'dh'
```

Positions are relative to the apex you collected or drew from, so a plan and a
sub-plan of it number differently, but a run and that run's drawing always agree.

## Reclaiming storage

**A store keeps everything you collect into it, until you delete the file.** matchlab
never removes an artifact on its own initiative. That's deliberate. An artifact's value
has nothing to do with whether your program still holds the variable that produced it.
The next process to rebuild the same plan wants a cache hit, not a rerun.

The cost of that is real, so every collect reports it. That's the `Store 3.0 MB (+3.0 MB),
7 artifacts` clause above. You can also ask directly:

```python
from matchlab import default_adapter

stats = default_adapter().stats()
print(stats.location)  # where the default store actually is
print(stats.bytes)  # what it costs, in bytes
print(stats.artifacts)  # {'source': 8, 'view': 40, 'model': 32, 'resolver': 24}
print(stats.describe())  # 'Store 1.2 GB, 104 artifacts'
```

Watch the artifact count rather than the size to see this happen. Edit a cleaning
expression, re-collect, and the count grows while the plan stays the same size. The old
artifacts are still there, and nothing will remove them.

Each adapter reports what only it can measure. A `DuckDBAdapter` hands back a
`DuckDBStoreStats`, with a `path` you can pass to `unlink()`, and a `free_bytes` for
space already freed inside the file.

### Trimming

`trim()` keeps what you name and deletes the rest:

```python
entities = build_plan().collect()

result = default_adapter().trim(keep=entities.fingerprints())
print(result.describe())
# 'Removed 80 artifacts, kept 24, reclaimed 416.2 MB'
```

`plan.fingerprints()` names every artifact a plan is made of, its own and its inputs'.
Which artifacts those are is the plan's business, not the store's, so the plan is what
answers. `keep` also takes the name of a published label, which keeps that resolution
and the sources it reads through:

```python
default_adapter().trim(keep=[*entities.fingerprints(), "production"])
```

**Published labels are kept whether or not you list them**, because publishing is the
strongest way this library has of saying "keep this". Losing one to a forgotten
argument would be indefensible. Trimming with nothing to keep and nothing published
raises, rather than emptying the store.

Nothing is inferred about what you are still using. matchlab does not watch which
objects your program is holding and treat the rest as rubbish. A store outlives the
process that wrote it, so what some interpreter happens to have in scope says nothing
about what is worth keeping. You say what to keep.

Trimming rewrites the store, which reopens its connection. Any session setting applied
through `adapter.conn` (see [Keeping memory bounded](#keeping-memory-bounded)) has to
be applied again afterwards.

### Starting from cold

Deleting the file is still the way to throw everything away:

```python
from pathlib import Path

Path("./run.duckdb").unlink()  # start again from cold
```

The default store lives in your user cache directory.
`default_adapter().stats().location` will tell you exactly where, and it's safe to
delete at any time. You lose cache hits, not results you can't rebuild, provided the
warehouse data hasn't moved.

!!! warning "DuckDB files do not shrink"
    Deleting rows or dropping tables inside a DuckDB file does **not** return space to
    the operating system. DuckDB marks the blocks free and reuses them for later
    writes, but the file stays the size of its high-water mark. There is no
    `VACUUM FULL`, and `CHECKPOINT` will not do it either:

    ```
    after write:               149.7 MB
    after DROP + CHECKPOINT:   149.7 MB
    ```

    This is why `trim()` **rewrites** the store rather than deleting inside it. Purging
    artifacts alone would buy reuse headroom while your disk usage stayed exactly the
    same. On a real 575 MB store, deleting 77% of its artifacts freed nothing at all.
    Copying what survives into a fresh file and swapping it in recovered 437 MB of that
    store, in half a second. It is the manual "collect what you want into a new store
    and delete the old one", done for you and without the re-collect.

    A trim reports what it actually recovered, measured before and after. It never
reports what it deleted. Those are different numbers, and only one of them is on

### Keeping memory bounded

An in-memory store (`DuckDBAdapter(":memory:")`) is not limited to RAM. DuckDB spills
table data to a temporary directory once it exceeds `memory_limit`. That defaults to
about 80% of your machine's memory:

```python
adapter = DuckDBAdapter(":memory:")
adapter.conn.execute("SET memory_limit = '4GB'")
adapter.conn.execute("SET temp_directory = '/fast/scratch'")
```

That bounds the resident footprint without discarding anything. That's almost always
what you want from a cache. Paged out is cheap to read back, deleted has to be
recomputed.

## Next

[Query the result :octicons-arrow-right-16:](./query.md){ .md-button .md-button--primary }
