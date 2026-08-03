# Glossary

A list of repository-specific term definitions, used consistently across matchlab's
code and documentation.

## Adapter

Where a plan's artifacts are stored and read back, keyed by [fingerprint](#fingerprint).
An adapter is storage, not an engine — it does not resolve anything itself.
`DuckDBAdapter` is the reference implementation, backing a single DuckDB database file
(or `:memory:`).

## Artifact

The stored output of one [step](#step): a source's extract and leaf assignment, a
view's materialised table, a model's edge list, or a resolver's complete resolution. A
[store](#store) keeps every artifact it is given until something explicitly
[trims](#trim) it.

## Collect

Run a plan. `collect()` walks a plan upstream-first and runs only the steps that aren't
already in the store, so re-collecting an unchanged plan does no work and adding a step
to a collected plan runs only the new one. See [content-addressed](#content-addressed).

## Content-addressed

Identified by what it is, not by when or how it was built. A step's
[fingerprint](#fingerprint) comes from its own configuration and its inputs'
fingerprints, so the same plan always keys the same artifact — collecting it twice
reads the cached one back rather than recomputing it.

## Entity

The real-world thing several records refer to. A [resolver](#resolver) groups records
under one [root](#root) ID when it decides they describe the same entity.

## Fingerprint

The 32-byte SHA-256 digest that keys a step's stored [artifact](#artifact). Two steps
with identical configuration and identical input fingerprints hash to the same
fingerprint, which is what makes a store [content-addressed](#content-addressed).

## Label

A pointer from a name someone chose to a resolution's [fingerprint](#fingerprint),
created by `publish()`. A label belongs to the store, not the plan — it can be re-aimed
at a different resolution, whereas a source's `name` is part of that source's own
output and never moves.

## Leaf

The stable ID of one record — a hash of its content, not its key. Identity coming from
content rather than key is why a leaf ID changes only when the underlying data does,
which is the basis matchlab anchors evaluation judgements to. Reading a source or view
directly, without going through a resolver, exposes the leaf as the `id` column.

## Merge-forward

The guarantee that a resolver's stored resolution carries forward every record
reachable from its inputs, not just the ones its own models formed edges over. A record
no model touches keeps its upstream grouping, or becomes a singleton, rather than
vanishing. matchlab's resolvers materialise this complete table once, at collect time,
rather than resolving on demand.

## Model

A step that scores candidate matches: `.dedupe()` within one view, or `.link()` between
two. A model produces edges, not clusters — turning edges into entities is a
[resolver's](#resolver) job.

## Plan

A tree of [steps](#step). Each step holds a reference to its own inputs, so the step
you are holding is the pipeline — there is no separate object to register steps with.
Nothing runs until you [collect](#collect) it.

## Position

A step's place in `collect()`'s run order, shown in brackets by `draw()` and quoted in
logs (`[step 5]`). Steps have no names, so position is how a log line, a drawn tree, and
`plan.lineage()` all refer to the same step. Positions are relative to the step a plan
or drawing starts from, so a sub-plan numbers its steps differently from the full plan
it came from.

## Resolver

A step that collapses a model's scored edges into clusters, one per [entity](#entity).
`.resolve()` defaults to connected components. A resolver's stored resolution is always
complete and [merge-forward](#merge-forward).

## Root

The ID of a cluster a resolver produces — a hash of the sorted set of [leaf](#leaf) IDs
it contains. Two runs that produce the same clustering produce the same root ID,
whatever order the underlying algorithm found its clusters in. Reading a view through a
resolver exposes the root as the `id` column, so several records can share one `id`.

## Source

Where a plan starts: a warehouse query, plus the column that keys it. Every column the
query returns is part of a record's identity, so two rows are the same record exactly
when the query returns identical values for both.

## Step

One node in a [plan](#plan): a source, view, model, or resolver. A step's kind decides
what it produces, and what [artifact](#artifact) a store keeps for it.

## Store

Where a plan's artifacts live once collected — the reference implementation is a
DuckDB database. A store keeps everything given to it until something explicitly
[trims](#trim) it or the file is deleted.

## Trim

Delete every artifact except the ones named or published, and reclaim the space that
frees. For a file-backed store this rewrites the file, because deleting rows inside it
does not return space to the operating system.

## View

A step that says which records a model matches over, and what shape they're in. Only
the columns named in its cleaning survive, plus `id` — the grouping the model matches
on.

