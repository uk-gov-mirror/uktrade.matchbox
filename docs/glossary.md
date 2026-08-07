# Glossary

A list of repository-specific term definitions, used consistently across matchlab's
code and documentation.

## Adapter

Where a plan's artifacts are stored and read back, keyed by [fingerprint](#fingerprint).
An adapter is storage, not an engine. It does not resolve anything itself.
`DuckDBAdapter` is the reference implementation, backing a single DuckDB database file
(or `:memory:`).

## Artifact

The stored output of one [step](#step), such as a source's [extract](#extract) and
leaf assignment, a view's materialised table, a model's edge list, or a
[resolver](#resolver)'s complete, merge-forward output. A [store](#store) keeps every
artifact it is given until something explicitly [trims](#trim) it.

## Collect

Run a plan. `collect()` walks a plan upstream-first and runs only the steps that aren't
already in the store, so re-collecting an unchanged plan does no work and adding a step
to a collected plan runs only the new one. See [content-addressed](#content-addressed).

## Content-addressed

Identified by what it is, not by when or how it was built. A step's
[fingerprint](#fingerprint) comes from its own configuration and its inputs'
fingerprints, so the same plan always keys the same artifact. Collecting it twice
reads the cached one back rather than recomputing it.

## Entity

The real-world thing several records refer to. A [resolver](#resolver) groups records
under one [root](#root) ID when it decides they describe the same entity.

## Extract

The rows a [source](#source)'s query returns, cached exactly as read so they can be
read back without another trip to the warehouse. A source's [fingerprint](#fingerprint)
folds in a hash of its extract, so a changed warehouse produces a new fingerprint
rather than silently reusing stale data.

## Fingerprint

The 32-byte SHA-256 digest that keys a step's stored [artifact](#artifact). Two steps
with identical configuration and identical input fingerprints hash to the same
fingerprint, which is what makes a store [content-addressed](#content-addressed).

## Judgement

A person's decision that one reviewed cluster's records do, or do not, describe the
same [entity](#entity), recorded by matchlab's evaluation tools and scored against a
[resolver](#resolver)'s clusters. A judgement is anchored to the content a reviewer was
shown, not to any record's key, which is why a [leaf](#leaf) ID, a hash of content,
rather than a key, is what a judgement endorses or rejects.

## Label

A pointer from a name someone chose to a [resolver](#resolver)'s
[fingerprint](#fingerprint), created by [publishing](#publish) a resolver's output. A
label belongs to the store, not the plan. It can be re-aimed at a different resolver,
whereas a source's `name` is part of that source's own output and never moves.

## Leaf

The stable ID of one record, a hash of its content rather than its key. Identity
coming from content rather than key is why a leaf ID changes only when the underlying
data does, which is the basis matchlab anchors evaluation judgements to. Reading a
source or view directly, without going through a resolver, exposes the leaf as the
`id` column.

## Merge-forward

The guarantee that a [resolver](#resolver)'s stored output carries forward every
record reachable from its inputs, not just the ones its own models formed edges over. A
record no model touches keeps its upstream grouping, or becomes a singleton, rather
than vanishing. matchlab's resolvers materialise this complete table once, at collect
time, rather than resolving on demand.

## Methodology

The matching algorithm a [model](#model) or [resolver](#resolver) step runs. Models
run `NaiveDeduper`, `DeterministicLinker`, `SplinkLinker`, or
`WeightedDeterministicLinker`. Resolvers run connected components. A model wraps one
methodology, chosen by `model_class`, using a **Deduper** to match within one view or
a **Linker** to match between two. Swapping the methodology only means changing
`model_class` (or `resolver_class`), never restructuring the plan around it.

## Model

A step that scores candidate matches by running one [methodology](#methodology),
either `.dedupe()` within one view or `.link()` between two. A model produces edges,
not clusters. Turning edges into entities is a [resolver's](#resolver) job.

## Plan

A tree of [steps](#step). Each step holds a reference to its own inputs, so the step
you are holding is the pipeline. There is no separate object to register steps with.
Nothing runs until you [collect](#collect) it.

## Position

Where something falls in an ordered sequence, used instead of a name. The repository
uses the word for two different sequences, and the numbers do not agree with each
other.

A step's place in `collect()`'s run order is one sense. `draw()` shows it in brackets,
and logs quote it (`[step 5]`). Steps have no names, so position is how a log line, a
drawn tree, and `plan.lineage()` all refer to the same step. Positions are relative to
the step a plan or drawing starts from, so a sub-plan numbers its steps differently
from the full plan it came from.

A setting that must point at one of a step's own inputs uses the other sense. It names
that input by its index among the step's own inputs, not by that input's place in the
whole plan. `ComponentsSettings.thresholds` keys a per-model threshold this way. A
model can sit at plan position 5 while still being resolver-input position 0.

## Publish

Point a [label](#label) at a [resolver](#resolver)'s output, so it can be found
without the plan that built it. Publishing is an act, not a property of the plan.
Nothing exists to point at until the resolver has been [collected](#collect), so
publishing always happens after collection, never before or instead of it.

Re-publishing the same label at the same resolver output is a no-op. Aiming an
existing label at a different resolver output needs `overwrite=True`, since that is
how you lose track of what a label used to mean.

## Qualify

Prefix a column name with its [source](#source)'s name (`first_name` becomes
`crn_first_name`), so the same field from two sources can sit side by side without
colliding. A qualified column's prefix must parse as a valid identifier, which is why a
source's `name` is restricted to safe characters.

## Resolver

A step that collapses a model's scored edges into clusters, one per [entity](#entity).
`.resolve()` defaults to connected components. A resolver's stored output is always
complete and [merge-forward](#merge-forward).

Call this a **Resolver**, or its **merge-forwarded Resolver output** if you mean the
stored table specifically. Don't call it "a resolution". That noun has been retired.

## Root

The ID of a cluster a resolver produces, a hash of the sorted set of [leaf](#leaf) IDs
it contains. Two runs that produce the same clustering produce the same root ID,
whatever order the underlying algorithm found its clusters in. Reading a view through a
resolver exposes the root as the `id` column, so several records can share one `id`.

## Source

The warehouse query a plan starts from, plus the column that keys it. Every column the
query returns is part of a record's identity, so two rows are the same record exactly
when the query returns identical values for both.

## Step

One node in a [plan](#plan). Steps are sources, views, models, or resolvers. A step's
kind decides what it produces, and what [artifact](#artifact) a store keeps for it.

## Store

Where a plan's artifacts live once collected. The reference implementation is a
DuckDB database. A store keeps everything given to it until something explicitly
[trims](#trim) it or the file is deleted.

## Trim

Delete every artifact except the ones named or [published](#publish), and reclaim the
space that frees. A file-backed store's trim rewrites the file itself, because
deleting rows inside it does not return space to the operating system.

## View

A step that says which records a model matches over, and what shape they're in. Only
the columns named in its cleaning survive, plus `id`, the grouping the model matches
on.

