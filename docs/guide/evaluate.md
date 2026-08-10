# Evaluate

Entity resolution has no single right answer. The same data supports many methodologies, each with many configurations, and the only way to choose between them is to measure.

matchlab treats that as a first-class job rather than something you bolt on afterwards.

## Sample clusters to judge

```python
from matchlab.eval import get_samples

samples = get_samples(n=20, resolver=companies)
```

`resolver` takes either a resolver you are holding, or the label one was [published](./build-a-plan.md) under. `review()` takes the same either/or. The label form needs no plan at all, which is what the rest of this page builds on.

Each sample is an `EvaluationItem`. It holds the records in one cluster, with their fields laid out so a human can see what was grouped and decide whether it should have been.

```python
item = samples[cluster_id]
item.records  # the rows, by leaf
item.get_unique_record_groups()  # identical rows collapsed
```

## Review clusters interactively

Sampling and judging by hand is fiddly. `review()` opens a terminal app that walks the queue for you. It shows one cluster on screen at a time, with records laid out so you can see what was grouped:

```python
from matchlab.eval import review

review(companies, tag="review-2026-07")
```

Paint the records into groups and each decision is stored as a judgement, tagged so a later `EvalData(adapter, tag=...)` scores against just this session. The app collects the resolver first if it isn't already.

To review the *same* clusters someone else was shown, agree on a seed:

```python
review(companies, seed=7)
```

Same store, same `n`, same seed, same clusters. Two people can judge the same work independently, and their judgements are directly comparable.

There's a command too, for when you'd rather not open a REPL. It takes a `module:attribute` naming a resolver in your own code. That's the same shape `uvicorn` uses:

```shell
matchlab review pipeline:companies --tag review-2026-07
```

`pipeline.py` is just your plan. The attribute can also be a function returning a resolver, if building it needs a warehouse connection you'd rather open on demand. Add `--log run.log` to keep logging off the screen.

### Reviewing a store on its own

You don't need the plan, or the warehouse, to review what it produced:

```shell
matchlab review companies --store ./run.duckdb
```

With `--store`, the target is the label a resolver output was **published** under, for example `companies.collect().publish("companies")`. This differs from naming Python code to import. It works because collecting a source caches its extract, and a stored resolver output records which source artifacts it covers. That's why the values on screen come out of the store.

This is also the more correct thing to review. It's the data the matching actually saw, not what the warehouse says today. It also means you can hand someone a `.duckdb` file, and they can judge it on a laptop with no database access.

```python
from matchlab.adapters import DuckDBAdapter
from matchlab.eval import review

review("companies", adapter=DuckDBAdapter("run.duckdb"), tag="second-opinion")
```

## Record a judgement

A judgement says *of the records you were shown, these belong together.* `review()` builds these for you. This is the same thing by hand, useful for scripting or a UI of your own.

```python
from matchlab.eval import create_judgement

judgement = create_judgement(
    item=item,
    assignments={0: "a", 1: "a", 2: "b"},  # group index -> group label
    tag="review-2026-07",
)

adapter.store_judgement(judgement, user_name="leo")
```

Records shown together but assigned to different groups are recorded as *negative* evidence, not just absence of positive evidence. This is what makes precision measurable rather than guessed.

## Score a resolver output

```python
from matchlab.eval import EvalData

evaluation = EvalData(adapter, tag="review-2026-07")
precision, recall = evaluation.precision_recall(companies)
```

Pass a resolver or the label one was published under. This is the same either/or as everywhere else on this page. Only pairs that appear in both the judgements and the resolver output are compared, so two methodologies scored against the same judgements are compared fairly.

## Comparing methodologies

The plan structure makes this cheap. Build several resolvers over the same sources and collect them. Because steps are content-addressed, the shared source is read once, and every candidate reuses it.

```python
naive = crn.dedupe(model_class=NaiveDeduper, ...).resolve().collect()
splink = crn.dedupe(model_class=SplinkLinker, ...).resolve().collect()
```

Then judge them **together**, in one pass:

```python
review([naive, splink], tag="bakeoff")

evaluation = EvalData(adapter, tag="bakeoff")
evaluation.precision_recall([naive, splink])
# [(0.91, 0.84), (0.95, 0.79)]
```

Handing `review()` several resolvers samples from their *merged* components. Two records appear together if either methodology put them there. That's what makes the comparison honest. Every cluster where the two could disagree is on screen, one judgement settles it for both, and neither gets to pick the clusters it's scored on.

Scoring them together matters for the same reason. `precision_recall` keeps only the pairs present in every resolver output *and* in the judgements, so each candidate is measured over the same records. Score them one at a time and each gets its own comparison set. Those numbers don't line up.

Publishing each one, for example `naive.publish("naive")`, is what lets you come back to it later with `matchlab review naive`. A candidate you only wanted to score once needs no label.
