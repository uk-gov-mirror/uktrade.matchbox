# Why matchlab

## Focus on your matching problem

matchlab gives you an off-the-shelf linker or deduper, so you don't have to implement your own matching logic. For example, it ships a deterministic linker with multiple rounds, and a weighted linker.

Good alternatives already exist for matching two datasets. Splink is a strong choice for probabilistic matching. The hard part is combining multiple levels of deduplication and linking across more than two datasets. matchlab lets you chain multiple Splink steps together, handling how data moves from one step to the next. It also implements algorithms like connected components efficiently, so you don't have to.

matchlab pays off where the work gets *harder*: more sources, resolved in layers, matched repeatedly, judged, and handed on.

All this frees up your time and attention to concentrate on what matters: cleaning your data, and deciding the rules for when two records describe the same entity.

## Avoid subtle mistakes

Moving between records and entities at each stage can quietly go wrong. A broken step often produces a plausible answer instead of an error. For example:

- **Fall-through**: records nothing matched must survive as singletons, not vanish.
- **Carrying records through a merge**: when a link joins two entities, every record inside them must come along. Skip one, and the earlier deduplication is silently undone.
- **Projection**: collapsing an entity to one row means deciding what its name *is*. Picking an arbitrary row's value is a silent data bug.

matchlab's resolver does all three (the merge-forward property), so you can trust its output to be correct by construction.

## Iterate fast

matchlab plans are lazy, so you can write them out and reason about them before anything runs. When you call `collect()`, it runs only the steps whose inputs changed, and re-collecting an unchanged plan does nothing. Matching *is* iteration: you change a cleaning rule, a threshold, a comparison, and look again. The cost that matters is the second run.

It's also easy to pivot your matching strategy. Changing the deterministic linker for Splink means changing `model_class` and `model_settings`. The rest of the plan doesn't move. By hand, adopting Splink means reshaping frames to fit its API, thresholding its scores, and feeding the result back into your clustering. matchlab runs the matcher. It doesn't replace it, so the same plan compares naive, deterministic, weighted, and probabilistic methods without restructuring. Methodology is a swap, not a rewrite.


## Evaluate systematically and honestly

You cannot improve a match you cannot measure, and "does this look right?" doesn't scale. matchlab treats measurement as part of the job, not a bolt-on: sample clusters, record judgements, score precision and recall, and compare two methodologies against the *same* judgements on equal terms. A terminal reviewer (`matchlab review`) handles the judging itself.

The subtle part is what a judgement is anchored to. matchlab identifies a record by a **leaf**, a hash of its content rather than its key, because a judgement is a decision made about the evidence a human actually saw, such as a name or a postcode. Anchor a judgement to the key and it outlives the evidence: a match decided on "acme / london" still stands after the row becomes "acme / manchester". That credits a decision to evidence nobody saw. Anchor it to content and the judgement decays exactly when its basis does. This is the difference between an evaluation trail you can trust months later and one that quietly drifts.

## Share your results

**The store is self-describing.** Hand someone a `.duckdb` file and they can review and score it with no warehouse access and no copy of your pipeline: `matchlab review entities --store run.duckdb`. They see the data the matching actually saw, which is also the more correct thing to judge against than whatever the warehouse says today.

matchlab can also export a resolver's output as a lookup file that translates and groups IDs across datasets. Write it back to the warehouse, and analysts can deduplicate and link data there directly, using the entities you've already resolved for them.


## Make your matching pipelines reproducible

All matchlab plans serialise to JSON, so you can export and reconstruct them quickly. You can share a plan with someone else without your personal configuration, such as how you connect to your warehouse. This also lets a plan travel across environments: build one in a notebook, save it to object storage, and let a scheduled job fetch and run it every day.