# Evaluate

Entity resolution has no single right answer. The same data supports many
methodologies, each with many configurations, and the only way to choose between them
is to measure.

matchlab treats that as a first-class job rather than something you bolt on afterwards.

## Sample clusters to judge

```python
from matchlab.eval import get_samples

samples = get_samples(n=20, resolution=entities)
```

`resolution` takes either a resolver you are holding or the label one was
[published](./build-a-plan.md) under — the same either/or that `review()` takes. The
label form needs no plan at all, which is what the rest of this page builds on.

Each sample is an `EvaluationItem`: the records in one cluster, with their fields laid
out so a human can see what was grouped and decide whether it should have been.

```python
item = samples[cluster_id]
item.records  # the rows, by leaf
item.get_unique_record_groups()  # identical rows collapsed
```

## Review clusters interactively

Sampling and judging by hand is fiddly. `review()` opens a terminal app that walks the
queue for you — one cluster on screen at a time, records laid out so you can see what
was grouped:

```python
from matchlab.eval import review

review(entities, tag="review-2026-07")
```

Paint the records into groups and each decision is stored as a judgement, tagged so a
later `EvalData(adapter, tag=...)` scores against just this session. The app collects
the resolver first if it isn't already.

To review the *same* clusters someone else was shown, hand it a dump:

```python
entities.get_matches().as_dump().write_parquet("sample.parquet")
review(entities, sample_file="sample.parquet")
```

A dump records which records appeared together, not their values, so the resolver is
still needed — the sources are re-read to display them.

There's a command too, for when you'd rather not open a REPL. It takes a
`module:attribute` naming a resolver in your own code — the same shape `uvicorn` uses:

```shell
matchlab review pipeline:entities --tag review-2026-07
```

`pipeline.py` is just your plan. The attribute can also be a function returning a
resolver, if building it needs a warehouse connection you'd rather open on demand. Add
`--log run.log` to keep logging off the screen.

### Reviewing a store on its own

You don't need the plan, or the warehouse, to review what it produced:

```shell
matchlab review entities --store ./run.duckdb
```

With `--store`, the target is the label a resolution was **published** under —
`entities.collect().publish("entities")` — rather than Python to import. This works
because collecting a source caches its extract, and
a stored resolution records which source artifacts it covers — so the values on screen
come out of the store.

That's also the more correct thing to review: it's the data the matching actually saw,
not what the warehouse says today. And it means you can hand someone a `.duckdb` file
and they can judge it on a laptop with no database access.

```python
from matchlab.adapters import DuckDBAdapter
from matchlab.eval import review

review("entities", adapter=DuckDBAdapter("run.duckdb"), tag="second-opinion")
```

## Record a judgement

A judgement says: *of the records you were shown, these belong together.* `review()`
builds these for you; this is the same thing by hand, for scripting or a UI of your
own.

```python
from matchlab.eval import create_judgement

judgement = create_judgement(
    item=item,
    assignments={0: "a", 1: "a", 2: "b"},  # group index -> group label
    tag="review-2026-07",
)

adapter.store_judgement(judgement, user_name="leo")
```

Records shown together but assigned to different groups are recorded as *negative*
evidence, not just absence of positive evidence — which is what makes precision
measurable rather than guessed.

## Score a resolution

```python
from matchlab.eval import EvalData

evaluation = EvalData(adapter, tag="review-2026-07")
precision, recall = evaluation.precision_recall(entities.results_eval())
```

Only pairs that appear in both the judgements and the resolution are compared, so two
methodologies scored against the same judgements are compared fairly.

## Measuring against known truth

When you generate data rather than judging it, you can assert against truth directly.
The testkit builds sources whose true entities are known:

```python
from matchlab.core.factories.scenarios import link_scenario

scenario = link_scenario(n_true_entities=100)
resolution = scenario.apex.collect().resolution()
```

`scenario.linked.true_entities` holds the planted entities, so you can compare the
resolved partition against them exactly. This is how matchlab tests itself, and it's
the fastest way to sanity-check a methodology before pointing it at real data.

See [`matchlab.core.factories`](../api/core/factories.md) for the generators.

## Comparing methodologies

The plan structure makes this cheap: build several resolvers over the same sources,
collect them, and score each against the same judgements.

```python
candidates = {
    "naive": crn.dedupe(model_class=NaiveDeduper, ...).resolve(),
    "splink": crn.dedupe(model_class=SplinkLinker, ...).resolve(),
}

for name, candidate in candidates.items():
    candidate.collect().publish(name)
    print(name, evaluation.precision_recall(candidate.results_eval()))
```

Publishing each one is what lets you come back to it later with
`matchlab review naive` — a candidate you only wanted to score once needs no label.

Because steps are content-addressed, the shared source is read once and both resolvers
reuse it.
