# Why matchlab

## Focus on your matching problem

Using matchlab means that you can use an off-the-shelf linker or deduper instead of having to implement your own bespoke logic. For example, matchlab provides a deterministic linker with multiple rounds, or a weighted linker.

There are great alternatives for matching two datasets, our favourite being Splink for probabilistic matching. The difficulty is writing logic that combines multiple levels of deduplication and linking across more than 2 datasets. You can use matchlab to weave together multiple Splink steps without having to think about how data needs to be processed from one to the next. We do the hard work behind the scenes, including implementing efficiently algorithms like connected components.

The case for matchlab is about what gets *harder* as the work
gets more serious: more sources, resolved in layers, matched repeatedly, judged, and
handed on.

All this frees up your time and brain power to concentrate on what matters: the cleaning of your data, and the definition of rules to decide when different records are the same entity.

## Avoid subtle mistakes

Mapping from entities to records at each stage has the potential to go quietly wrong, with attempts producing a plausible answer rather than an error, for example:

- **Fall-through** — records nothing matched must survive as singletons, not vanish.
- **Carrying records through a merge** — when a link joins two entities, every record
  inside them has to come along, or the earlier deduplication is silently undone.
- **Projection** — collapsing an entity to one row means deciding what its name *is*;
  picking an arbitrary row's value is a silent data bug.

matchlab's resolver does all three (the "merge-forward" property), so you can trust resolution to be correct by construction.

## Iterate fast

matchlab plans are lazy, so you can write them out and reason about them before any processing starts to happen. When you finally `collect()` only the steps whose inputs changed are run, and re-collecting an unchanged plan does nothing. Matching *is* iteration —
you change a cleaning rule, a threshold, a comparison, and look again — so the cost that
matters is the second run.

It's also easy to pivot your matching strategy. Changing the deterministic linker for Splink is `model_class` and `model_settings` —
the plan around it doesn't move. By hand, adopting Splink means reshaping frames for its
`unique_id` convention, thresholding its scores, and feeding the result back into your
clustering. matchlab runs the matcher; it doesn't replace it, so the same plan compares naive, deterministic, weighted and probabilistic methods without restructuring. Methodology is a swap, not a rewrite.


## Evaluate systematically and honestly

You cannot improve a match you cannot measure, and "does this look right?" doesn't scale. matchlab treats measurement as part of the job, not a bolt-on:
sample clusters, record judgements, score precision and recall, and compare two
methodologies against the *same* judgements on equal terms — with a terminal reviewer
(`matchlab review`) for the judging itself.

The subtle part is what a judgement is anchored to. A record's identity is a hash of its
**content**, not its key, because a judgement is a decision made about the evidence a
human actually saw — a name, a postcode. Anchor a judgement to the key and it outlives
the evidence: a match decided on "acme / london" still stands after the row becomes
"acme / manchester", a decision credited to evidence nobody saw. Anchor it to content and
the judgement decays exactly when its basis does. This is the difference between an
evaluation trail you can trust months later and one that quietly drifts.

## Share your results

**The store is self-describing.** Hand someone a `.duckdb` file and they can review and
score it with no warehouse access and no copy of your pipeline — `matchlab review
entities --store run.duckdb`. They see what the data the matching actually saw, which is also
the more correct thing to judge against than whatever the warehouse says today.

matchlab also simplifies the export of a resolution as a lookup file translating and grouping IDs from different datasets. This can be written back to a data warehouse to allow analysts to easily deduplicate and link data from the warehouse that you have resolved for them.


## Make your matching pipelines reproducible

All matchlab plans serialise to JSON, which means you can export them and reconstruct them in a jiffy. This lets you easily share your plans with others without the configuration that's specific to you, like how to connect to your data warehouse. This also allows plans to travel across environments: you could formulate a plan in a notebook, and save it to object storage so a scheduled job can fetch it and run it every day.