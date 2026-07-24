# Four sources, two ways

The same pipeline written twice: once with matchlab, once with Polars and a
disjoint set. Both dedupe four company sources, link the deduplicated entities to each
other, and print the resulting groups.

```shell
python warehouse.py       # build the SQLite warehouse both read
python with_matchlab.py
python by_hand.py         # identical output
```

`by_hand.py` is meant to be a fair opponent, not a strawman. It uses the same matching
rule, idiomatic Polars, and a union-find that handles the cases that matter. A test
(`test/test_examples.py`) asserts the two agree, so this comparison can't quietly rot.

## What it costs

The 21-record example is for reading. For timing, `benchmark.py` generates the same
shape of data at any scale — duplicates within sources, overlap between them — and runs
both pipelines over it:

```shell
python benchmark.py 10_000 100_000 1_000_000 5_000_000
```

| rows | by hand | matchlab | ratio | matchlab re-run |
|---:|---:|---:|---:|---:|
| 13 k | 0.05 s | 0.40 s | 8.0× | 0.04 s |
| 130 k | 0.40 s | 1.24 s | 3.1× | 0.21 s |
| 1.3 M | 4.16 s | 12.06 s | 2.9× | 2.21 s |
| 6.5 M | 21.6 s | 67.6 s | 3.1× | 12.0 s |

Both scale linearly, and **matchlab costs a stable ~3× once the data is big enough to
matter**. The eye-catching multiple at tiny sizes is fixed overhead — a DuckDB store,
content-hashing every row, fingerprinting every step — and it amortises away. Don't
draw conclusions from a 21-row pipeline in either direction.

Three things worth reading off that table:

- **3× is the honest price**, and it buys something: a queryable store, a complete flat
  resolution, and caching. It is not a scaling problem.
- **The re-run column beats `by_hand` outright.** From the second run on, matchlab does
  2.2 s of nothing where the hand-rolled pipeline does 4.2 s of work again. If you
  iterate — and matching is iteration — that flips the comparison.
- **Both produce identical groups at every size**, which is what makes the rest of this
  README worth reading.

Lines of pipeline code: 65 against 111. Real, but not the argument either.

## What isn't in `by_hand.py` at all

```python
entities.lookup_key(from_source="crn", to_sources=["dh", "cdms", "hmrc"], key="crn-01")
# {'crn': ['crn-01', 'crn-02'], 'dh': ['dh-100'], 'cdms': ['cdms-A'], 'hmrc': ['hmrc-9001']}
```

Also absent: sampling clusters for review, recording judgements, scoring precision and
recall against them, the review TUI, and a store you can hand to a colleague who has no
warehouse access. Each is buildable. Each is more code than the pipeline.

## Where the honest answer is "just use Splink"

**Two datasets, one link, run once.** Splink does the matching and gives you clustering.
matchlab adds a plan, a store and vocabulary for no return.

**You need probabilistic matching and nothing else.** matchlab doesn't do matching — it
*runs* Splink. If Splink alone solves your problem, that's the whole answer.

**One run, and throughput is the whole job.** 3× is 3×. If you resolve once and never
look again, that multiplier buys you nothing you'll use.

## Where it stops being the answer

**More than two sources, resolved in layers.** The entity/record juggling above is
per-layer, and it compounds.

**You'll change it.** Swapping the deterministic matcher for Splink here is
`model_class` and `model_settings`. By hand it means reshaping frames for Splink's
`unique_id` convention, thresholding its scores, and feeding them back into the union
find — at both the dedupe and the link level.

**You need to know if it got better.** This example gets 9 groups for 8 real companies:
`dh-105` "Theta Retail" and `hmrc-9003` "Theta Retail Group Ltd" clean to different
names, so exact matching can't join them. Finding that out is what the evaluation tools
are for, and it's the thing neither file does by matching harder.
