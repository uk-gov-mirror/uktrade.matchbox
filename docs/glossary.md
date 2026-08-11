# Glossary

A list of repository-specific term definitions, used consistently across matchlab's
code and documentation.

## Adapter

Where a plan's artifacts are stored and read back, keyed by [fingerprint](#fingerprint).
An adapter is storage, not an engine. It does not resolve anything itself.
`DuckDBAdapter` is the reference implementation, backing a single DuckDB database file
(or `:memory:`).

## Answer key

🧪 Testkits

A [testkit's](#testkit) map from a generated row's own values to the
[true entity](#true-entity) that produced it. It exists because matchlab derives a
row's real ID by content-hashing at collect time, so no ID is knowable when the answer
key is built. A [matcher](#matcher) built from one reads a row's values back out of
whatever frame it is handed, and looks up the entity that row belongs to, which is what
lets it score a plan it has never seen. `AnswerKey` is the class.

## Artifact

The stored output of one [step](#step), such as a source's [extract](#extract) and
leaf assignment, a [transform](#transform)'s materialised frame, a model's edge list, or
a [resolver](#resolver)'s complete, merge-forward output. A [store](#store) keeps every
artifact it is given until something explicitly [trims](#trim) it.

## Cluster

🧪 Testkits

In a [testkit](#testkit), a set of records claimed to be one entity, an answer rather
than the answer. It carries only membership, which is what an actual model or resolver
result can be compared against. Contrast [true entity](#true-entity), which is the
planted answer a cluster is checked against. `Cluster` is the class.

## Collect

Run a plan. `collect()` walks a plan upstream-first and runs only the steps that aren't
already in the store, so re-collecting an unchanged plan does no work and adding a step
to a collected plan runs only the new one. See [content-addressed](#content-addressed).

## Content-addressed

Identified by what it is, not by when or how it was built. A step's
[fingerprint](#fingerprint) comes from its own configuration and its inputs'
fingerprints, so the same plan always keys the same artifact. Collecting it twice
reads the cached one back rather than recomputing it.

## Deduper

A [methodology](#methodology) that finds candidate duplicate pairs within one
[frame](#frame), run by `.dedupe()`. `NaiveDeduper` is the built-in. A custom one
subclasses `Deduper` and registers with `add_model_class`, the same pattern
[transformers](#transform) plug in with.

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

## Frame

The records a [model](#model) matches over, a table carrying an `id`. A
[source](#source) (`id` is the leaf), a [transform](#transform), and a
[resolver](#resolver) (`id` is the entity root, which is what makes layering possible)
are the three kinds. Every frame chains the same verbs (`select`, `clean`, `group`,
`transform`, `dedupe`, `link`). A [model](#model) is the one step that is not a frame,
since it yields edges rather than records.

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
source or frame directly, without going through a resolver, exposes the leaf as the
`id` column.

## Linker

A [methodology](#methodology) that finds candidate matches between two
[frames](#frame), run by `.link()`. `DeterministicLinker`, `WeightedDeterministicLinker`,
and `SplinkLinker` are the built-ins. A custom one subclasses `Linker` and registers
with `add_model_class`.

## Location

matchlab's handle on the warehouse a [source](#source) reads from: a database, plus the
client that reaches it. `RelationalDBLocation` is the reference implementation, wrapping
a SQLAlchemy or ADBC client. The warehouse is the external database itself; the location
is the handle, and it is where a source's rows come *from* — distinct from the
[store](#store), where a plan's [artifacts](#artifact) are kept once collected. A
location resolves nothing and stores nothing; it only answers a query.

## Matcher

🧪 Testkits

A model in a [testkit](#testkit) that already knows the answer, built from an
[answer key](#answer-key) instead of scoring anything. `PerfectDeduper` and
`PerfectLinker` are the two matchers. A matcher makes no mistakes, so pointing one at
generated data leaves the plan as the only variable under test. Build one through
`LinkedSources.dedupe()` or `.link()` rather than directly.

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
methodology, chosen by `model_class`, using a [deduper](#deduper) to match within one
frame or a [linker](#linker) to match between two. Swapping the methodology only means
changing `model_class` (or `resolver_class`), never restructuring the plan around it.

## Model

A step that scores candidate matches by running one [methodology](#methodology). A
[deduper](#deduper) matches within one [frame](#frame) via `.dedupe()`, and a
[linker](#linker) matches between two via `.link()`. A model produces edges, not
clusters. Turning edges into entities is a [resolver's](#resolver) job.

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
whatever order the underlying algorithm found its clusters in. Reading a [frame](#frame)
through a resolver exposes the root as the `id` column, so several records can share
one `id`.

## Source

The warehouse query a plan starts from, plus the column that keys it. Every column the
query returns is part of a record's identity, so two rows are the same record exactly
when the query returns identical values for both.

## Step

One node in a [plan](#plan). Steps are sources, transforms, models, or resolvers. A
step's kind decides what it produces, and what [artifact](#artifact) a store keeps for
it.

## Store

Where a plan's artifacts live once collected. The reference implementation is a
DuckDB database. A store keeps everything given to it until something explicitly
[trims](#trim) it or the file is deleted.

## Testkit

🧪 Testkits

An object that generates test data with a known answer, and exposes what it knows
about that data. `matchlab.testkit` builds these. `GeneratedSource`, `GeneratedModel`,
and `LinkedSources` each wrap a real plan node, but expose only the fixture, meaning
its rows, its features, and its planted answer. Anything you do to the plan itself
goes through the node a testkit wraps, never through a forwarding shortcut on the
testkit.

A testkit is what makes the oracle testing pattern possible here. Assert against a
known, planted answer rather than an answer you have to work out by hand or guess at.

The package checks a plan's answer two ways, because matchlab derives a record's real
ID by content-hashing at collect time, so no ID is knowable when a fixture is built.
One check scores a [methodology](#methodology) directly, against the testkit's own
synthetic IDs, without collecting anything. The other scores a collected plan, keyed
by row values instead, through an [answer key](#answer-key). See `matchlab.testkit`'s
module docstring for which to use.

## Transform

A step that reshapes a [frame](#frame) with a pluggable, serialisable **transformer**.
`select` keeps only the named columns, `clean` derives new ones with DuckDB SQL while
keeping the rest, and `group` collapses each `id` to one row. `transform()` is the
general verb they desugar to (`.clean(...)` is `.transform(Clean(...))`).
`Transformer` is the base class. `Select`, `Clean`, and `Group` are the built-ins, and
a custom one registers with `add_transformer_class`, the same pattern
[dedupers](#deduper) and [linkers](#linker) plug in with. Its configuration is part of
the plan and its [fingerprint](#fingerprint), like any [methodology](#methodology)'s
settings.

## Trim

Delete every artifact except the ones named or [published](#publish), and reclaim the
space that frees. A file-backed store's trim rewrites the file itself, because
deleting rows inside it does not return space to the operating system.

## True entity

🧪 Testkits

The planted answer a [testkit](#testkit) generated data from. It is one real-world
thing, identified by the values it was generated from, spanning every source it landed
in. Project it onto the sources under test to get a [cluster](#cluster), the only form
that is comparable with an actual result. `TrueEntity` is the class.

