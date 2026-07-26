# `matchlab`: plan of attack

Turning Matchbox into a local-only Python library (`matchbox-db` → `matchlab`).

## The crux (validated)

The server is not a REST wrapper — it's the entity-resolution *engine*.
`server/postgresql/utils/query.py:_build_unified_query` stores the resolution graph
normalized and re-projects source keys up the hierarchy **on demand**, with a
priority-`COALESCE`, at every query. That design pays off only under server constraints
(many clients, immutable shared runs, precious storage) — none of which hold locally.

So we invert it: **materialise the complete resolution forward at collect time; never
resolve on demand.** The recursive query engine is *deleted, not ported*.

**Phase 0 finding (load-bearing).** A collected resolver must persist
`merge(upstream complete resolution, its own clusters)` — not just its own clusters.
Leaves grouped upstream but untouched downstream ("fall-through") must inherit their
upstream cluster, or they collapse to singletons. This is the eager equivalent of the
server's `COALESCE`. Proven in `spikes/phase0_materialize_forward.py`.

---

!!! note "Node names changed after Phase C"
    `Clean` → `Cleaner` → **`View`** (`views.py`), `Resolve` → **`Resolver`**. The
    Phase A/B/C sections below are a record of what was done at the time and keep the
    names current then; everything above and in "Phase D" uses the names in the code.

## Architecture: steps as a plan tree

### How Polars structures it

A `LazyFrame` wraps a logical plan node (`DslPlan`) whose variants *box their inputs*:
`Select { input, exprs }`, `Join { input_left, input_right }`, `Scan { … }` (leaf).

* **Edges point upstream only.** A node holds its inputs; there is no parent pointer and
  no consumer list.
* **No registry.** "The DAG" is whatever is reachable from the node you `.collect()`.
* **Nodes are immutable values.** Each op wraps the previous plan and returns a new
  handle; the old one stays valid. Shared subplans are one object (structural sharing).
* **`.collect()`** walks the tree, optimises (pushdown, common-subplan elimination),
  executes.

The lesson: **you need lineage, but not a container for it.** Lineage *is* the input
references on each node.

### The step model

```python
class Step(ABC):
    upstream: tuple[Step, ...]  # direct inputs ONLY — no downstream, no registry
    _fp: bytes | None  # memoised: set once collected
```

Steps carry no name. They are identified by **position** in `lineage.walk`; a name is
a pointer from the store to a resolution, set by `Resolver.publish`. `Source` is the
exception, since its name prefixes its columns and so is part of its output.

| Node | `upstream` |
|---|---|
| `Source` | `()` — a leaf, like `scan_parquet` |
| `View` | `(source…, resolver?)` |
| `Model` (dedupe / link) | `(left_view, right_view?)` |
| `Resolver` | `(model…)` |

Everything the old `DAG` did becomes a **derived walk from a root node**:
`node.lineage()`, `node.draw()`, `node.collect(adapter=None)`.
The `DAG` class disappears; what survives is `lineage.py` — **pure functions over a root
node** (`walk`, `number`, `draw`), deduplicating shared nodes by object identity.
Cycles need no validating: a step's inputs must exist before it does.

### Why the registry had to go

`DAG` was a mutable, name-keyed registry every step pointed back at. That one decision
caused four distinct problems:

| Problem | Registry cause |
|---|---|
| Circular import in `sources.py` | `Source` had to import `DAG` to create/join one, while `dags.py` imports `Source` — only because a node must be *registered* somewhere. |
| Steps could see downstream | `step.dag.nodes` exposes descendants. Semantically wrong for a lazy plan. |
| "Cannot mix DAGs" | Two independently-built sources sit in different registries. Polars has no such error because there is nothing to mix. |
| **GC could never fire** | `DAG.nodes` strong-references every step forever, so no node is ever unreachable. The registry made the brief's "clear data on garbage collection" *impossible*. |

### Design rules

1. **All inter-step data flows through the adapter — never through in-memory
   attributes.** A resolver reads its inputs' edges by `_fp`, and recomputes its upstream
   `(id, source, key, leaf)` from stored source leaves + the upstream resolver's stored
   resolution. Consequence: `low_memory`, `cache_leaf_ids`, `clear_data` and in-memory
   `results`/`_upstream` all **delete**. (`low_memory` is currently worse than redundant
   — it breaks resolvers, because materialising needs the query cache it drops.)
2. **One canonical verb per operation.** `clean` / `dedupe` / `link` / `resolve` /
   `collect`. Delete the nouns and run machinery: `query`, `deduper`, `linker`,
   `resolver`, `run_and_sync`, `sync`, `run`, `new_run`, `load_default`, `set_default`,
   `set_client`, `final_steps`, `default_resolver`. Apex detection is meaningless when
   you collect the node you are holding; `lookup_key` and `get_matches` move onto
   `Resolver`.
3. **`collect()` returns the same node, memoised.** Lazy until collected, cached after.
   A node is therefore both "lazy" and "collected" — satisfying *"each step can take as
   input either a lazy step or a collected one"* with no second type.
   `get_matches()` on an uncollected node collects first.
4. **`View` is a plan node, fused by default.** Not materialised unless you
   `.collect()` it — the equivalent of Polars' `.cache()`. Collecting it is the
   "show me my cleaned data" debugging path; once materialised, downstream reads it
   instead of re-fusing.
5. **Adapter: module default + per-collect override.** A lazily-created DuckDB store in
   the user cache dir; `set_default_adapter(...)` globally, `collect(adapter=...)` per
   call.
6. **GC by mark-and-sweep over a `WeakSet` of live steps.** Weak refs don't pin, so
   Python reachability drives lifetime. ⚠️ Content-addressing means two distinct nodes
   can share a fingerprint, so GC must sweep against the live set — naive per-node
   finalizer deletes would destroy a sibling's data.

---

## Target module layout

```
matchlab/
  __init__.py                  # Source, verbs, set_default_adapter
  steps.py                     # Step ABC + node types' shared behaviour
  lineage.py                   # pure functions over a root node (topo, walk, draw)
  sources.py                   # Source — a leaf; imports no plan machinery
  document.py                  # dump/load — a plan as a portable, derived document
  views.py  models.py  resolvers.py  results.py  locations.py
  adapters/                    # base.py (ABC) + duckdb.py — storage only
  core/                        # was matchbox.common (survivors)
    arrow.py  datatypes.py  hash.py  dsu.py  logging.py
    resolution.py              # materialise_resolution, leaf_id, root_id
    config.py                  # config models rescued from dtos.py
    eval.py  factories/
  eval/  cli/                  # cli shrunk: run a pipeline, eval TUI
```

Deleted wholesale: `matchbox/server/**`, `client/_handler/**`, `client/_settings.py`,
`client/dags.py`, server exceptions, API-transport DTOs,
`server/postgresql/utils/{query,insert}.py`.

**DTO split:** `dtos.py` mixes *transport* objects (`Collection`, `Run`, `UploadStage`,
`StepPath`, permissions → **deleted**) with *config* models (`SourceConfig`,
`ModelConfig`, `QueryConfig`, `ResolverConfig`, `Step`). The config models survive in
`core/config.py` without HTTP baggage.

They are not, though, "the serialisable DAG" — that was the original framing and it
was half right. A config is what a fingerprint hashes, so it must carry a step's
settings and *not* its edges; a serialisable DAG needs the edges. `document.py` holds
the other half: `PlanDocument` carries the nodes and the edges between them, each node
holding the config unchanged. See "Configs describe settings; documents describe
plans" below.

## Adapter contract (storage, not an engine)

```python
class Adapter(ABC):
    def has(fp) -> bool
    def store_source(fp, name, extract, leaves);  read_source_extract/leaves(fp)
    def store_model(fp, edges);                    read_model(fp)
    # resolution = merge(upstream complete, own clusters) — the Phase 0 finding
    def store_resolver(fp, resolution);            read_resolver(fp)
    def store_clean(fp, table);                    read_clean(fp)      # Phase A addition
    def store_judgement(...); read_eval_data(tag); sample(fp, n, seed)
    def gc(live: set[fp]) -> int;                  close()
```

## What each collected step persists

| Step | Artifact (fingerprint-keyed, namespaced by step kind) |
|---|---|
| `Source` | warehouse extract + `key → leaf_id` |
| `View` | *nothing by default* (fused); the view's table if explicitly collected |
| `Model` | edge list `(left_id, right_id, score)` |
| `Resolver` | complete flat resolution `(root, leaf, key, source)` |

Terminal reads: `query`/`match` **deleted**; `lookup_key`, `get_matches`,
`as_lookup` are filters/joins on the stored resolution.

---

## Status

**All phases complete. Outstanding: cut the first release (Phase C), and the two items
listed at the end of Phase D.**

Paths in the phase notes below are as they were at the time — `matchbox.*` before
Phase B, `common/` before it became `core/`.

**Done and durable — untouched by the re-architecture:**

* **Phase 0 ✅** — materialise-forward spike + equivalence oracle
  (`spikes/phase0_materialize_forward.py`, 3 tests). Surfaced the merge-forward finding.
* **Phase 1 ✅** — DuckDB adapter, storage only (`src/matchbox/adapters/`,
  `test/adapters/test_duckdb.py`, 15 tests) incl. a real eval round-trip through
  `precision_recall`.
* **Phase 2 ✅** — client runs fully local, no server. Client-side ID + resolution
  engine (`common/resolution.py`, `test/common/test_resolution.py`, 8 tests);
  `_handler` gone from the client; eval on the adapter; `import matchbox` needs no
  server env. Verified end-to-end on a real SQLite warehouse
  (`test/integration/test_local_dag.py`).
* **Phase 3 (partial)** — lazy `collect()` with incremental caching, lineage-scoped
  `lineage`/`draw`, fluent verbs and implicit DAG. **The *semantics* are validated and
  keep; the *implementation* is superseded by Phase A**, which moves them off the DAG
  registry onto the plan tree.

Current local suite: **324 passing, 0 skipped** (2026-07-24).

---

## Phase A — plan-tree re-architecture ✅ done

1. **`Step` + lineage ✅.** `Step` ABC (`name`, `upstream`, `_fp`) in `steps.py`;
   `lineage.py` holds `walk` / `find` / `validate` / `draw` as pure functions over a
   root node, deduplicating shared nodes by object identity.
2. **Node types ✅.** `Source` (leaf), `Clean` (fused), `Model`, `Resolve` — each
   holding only its inputs. `dags.py` and `queries.py` deleted.
3. **Adapter plumbing ✅.** `set_default_adapter` / `default_adapter` +
   `collect(adapter=...)`; `store_clean`/`read_clean` added to the adapter.
4. **Adapter-only data flow ✅.** `Resolve` reads its inputs' edges by `_fp` and rebuilds
   upstream identifiers from stored source leaves / resolutions. `low_memory`,
   `cache_leaf_ids`, `clear_data` and the in-memory `results`/`_upstream` are gone.
5. **Single verb set ✅.** `clean` / `dedupe` / `link` / `resolve` / `collect`;
   `get_matches` + `lookup_key` on `Resolve`. All `DAG`, `run_and_sync`, `sync`, `run`,
   run-lifecycle and noun methods deleted.
6. **GC ✅.** Module `WeakSet` of live steps; `gc()` mark-and-sweeps by live fingerprint
   (not per-node finalizers — content-addressing means distinct nodes can share one).
7. **Tests ✅.** `test/plan/test_lineage.py` (8, pure graph semantics with fake steps)
   and `test/plan/test_plan.py` (15, e2e over SQLite). The old `test/integration` suite
   is replaced.

**Improvement over the earlier design:** fingerprints are now derived from the *plan*
(kind + config + inputs' fingerprints), so `collect` checks the cache **before** running
a step rather than after. Sources stay data-fingerprinted — raw data enters there — so a
fresh `Source` re-reads the warehouse (the documented way to refresh) while an existing
one memoises. Cross-session caching now works: rebuild the same plan in a new process
and, if the warehouse is unchanged, everything downstream is a cache hit.

**Exit ✅:** no registry, no circular import, plan built with no DAG, and dropping a plan
reclaims its storage (`test_gc_reclaims_a_dropped_plan`). Local suite: 49 passing.

**Casualty list for Phase B:** `common/factories/**` and `client/cli/eval/**` still
import the deleted `dags`/`queries`. Both were already server-coupled and unusable
locally; Phase B removes them along with the server.

## Phase B0 — test & testkit salvage ✅ done

A survey of the 19,219-line suite showed the "delete the legacy tests" line in Phase B
is too blunt: a meaningful slice tests logic that *survives*. Salvage while the old code
is still present as a reference, then delete — the same principle as the Phase 0 oracle.

**Triage (measured, not estimated):**

| Category | Lines | Contents |
|---|---:|---|
| Delete | ~7,300 | `test/server/**`, `test/e2e/**`, `test/fixtures/{db,client}.py`, `test/scripts/**` |
| Keep as-is | ~2,300 | matching methodologies, components resolver, `test_locations`, `test_hash`, `test_datatypes`, `test_eval` |
| Port — cheap | ~1,400 | `test_results.py`, the `test_clean_*` block of `test_queries.py`, `test_sources.py` field/fetch/hash |
| Port — testkit | ~7,400 | `factories/**` (4,129) + factory tests (3,328) |

`test/client/test_dags.py` (1,615) is mostly superseded by `test/plan/test_lineage.py`;
skim for orphan semantics, then delete.

**Coverage gap introduced by Phase A (fix first):** the new plan suite covers none of
`combine_type` (concat/explode/set_agg), the cleaning SQL semantics (nine `test_clean_*`
tests today), or `ResolverMatches.view_cluster` / `.merge` — even though all three are
carried-over pure functions. These port nearly verbatim.

**Testkit decision — keep ground truth + generators.** The coupling is not uniform:
* `entities.py` (614) imports nothing from client or server. It is the ground-truth
  machinery — synthetic entities with known true clusters — and ports **free**. It is
  what lets tests assert precision/recall against truth rather than eyeball fixtures,
  which the brief names as a core differentiator.
* `sources.py` / `models.py` / `resolvers.py` (2,162): generation logic is pure; only the
  DAG/Query-binding wrappers need rewriting.
* `scenarios.py` (1,292) + `dags.py` (60): the only server-shaped part (`scenarios.py`
  imports `matchbox.server.base`). It is large *because* it drives a server — create
  collection, run, step, upload, set default. In the plan world a scenario is
  `crn.dedupe(...).resolve().collect()`, so this is a **net deletion**, not a port.

**Steps:**

1. **Cheap wins ✅.** `test/plan/test_cleaning.py` (11) ports the `test_clean_*` block.
   Porting it caught a **regression from Phase A**: `_apply_cleaning` used
   `if not cleaning`, conflating `None` (pass through) with `{}` (a real projection
   selecting no columns → identifiers only). Fixed and pinned.
2. **`entities.py` ✅ — zero changes needed.** It imports nothing from client or server;
   the ground-truth machinery ported free, exactly as predicted.
3. **`sources.py` ✅.** The DAG coupling was shallow and is gone: the threaded `dag`
   parameter, the `if dag is None: DAG("collection")` blocks, `Source(dag=...)`, and
   `LinkedSourcesTestkit.dag`. DAG-bound helpers replaced — `query()` → `clean()`;
   `path`, `fake_run()`, `into_dag()` deleted (no step paths, no in-memory results, no
   DAG to detach from). **Validation: all 52 existing source/linked/entity factory
   tests pass with no test changes.** Also surfaced a second Phase A regression —
   `Source.infer_types` had flipped its default to `True`, breaking `SourceField`
   callers; it now auto-detects from the argument type.
   `sqlite_in_memory_warehouse` moved from the server-coupled `test/fixtures/db.py`
   into `test/conftest.py`, starting the fixture detangle.
4. **`models.py` / `resolvers.py` ✅ ported — but surfaced a design gap.**
   `MockDeduper`/`MockLinker` are replaced by `ScriptedDeduper`/`ScriptedLinker`, which
   **return** the pre-generated scores, so collecting runs the real execution path;
   `fake_run()` and `into_dag()` are deleted rather than ported. Scores live in a
   content-addressed registry (`register_script`) keyed by hash, because settings are
   JSON-serialised into the step fingerprint and a DataFrame is not — keying by content
   keeps the fingerprint honest. `left_query`/`right_query` → `left`/`right` `Clean`
   nodes; `query_to_model_factory` → `clean_to_model_factory`. Testkits now build real
   plan nodes (`lineage: crn → clean_crn → model → resolve_model`) and collect
   end-to-end: 48 generated scores → 48 stored edges.

   ⚠️ **Finding — pre-generated scores were in the wrong ID space.** The first port
   replayed fixed edges, but they referenced the testkit's *synthetic* entity IDs while
   the plan's `Clean` emits *content-derived leaf IDs* computed at collect time.
   Measured overlap: **zero** — so nothing clustered (32 rows → 32 clusters). This
   worked before only because the server assigned IDs and the testkit both generated
   and uploaded the matching `data` table, controlling both sides.

5. **Value-grouping design ✅.** Ground truth is now keyed by row **values**, not IDs.
   `ValueTruth` maps a tuple of column values → true-entity ID (held per side, so a
   linker joins two sources into one entity space); `ScriptedDeduper`/`ScriptedLinker`
   read those columns out of the frame they are handed and emit edges between whatever
   IDs are actually present. `truth_from_testkits` derives it from generated sources.
   The testkit is therefore **independent of how identity is assigned** and survives
   future changes to ID minting. Result: 32 records → **8 clusters from 8 planted
   entities**.

6. **Ground-truth tests ✅.** `test/plan/test_ground_truth.py` (4): dedupe and
   cross-source link both assert the resolved partition **equals** the planted entities
   exactly, plus completeness (no record dropped) and precision (no cluster merges two
   entities). This is the ER-evaluation capability the brief names as a differentiator,
   and the reason the testkit was worth porting rather than deleting.

7. **Remaining tests ported ✅.** `test_model_factory.py` (20) and
   `test_resolver_factory.py` (5) run on the plan API. `clean_to_model_factory` was
   *deleted* rather than ported — its only surviving callers were `scenarios.py` and
   the server tests, so `model_factory` is now the single entry point (−103 lines).
   `Model.config` was not reintroduced: the three call sites only wanted
   `config.type`, which the plan node exposes as `model_type`. Resolver tests for
   `into_dag` / `fake_run` / DAG-detachment are gone with the machinery they tested.

8. **`scenarios.py` rewritten, `dags.py` deleted ✅.** As predicted, a large net
   deletion: **1,292 → 129 lines**. A scenario used to mean driving a server (create
   collection → run → step → upload → set default); now it is "generate linked
   sources, write them to a warehouse, wire up the nodes". `source_scenario`,
   `dedupe_scenario` and `link_scenario` each return a `Scenario` carrying both the
   plan and the truth it was built from. `TestkitDAG` has no meaning when the plan node
   *is* the container.

**Exit ✅.** Everything worth keeping runs on the plan API: **254 tests passing, no
collection errors, lint clean**. Phase B can now delete freely.

Net effect of B0: two Phase A regressions caught and fixed (`cleaning` None-vs-`{}`,
`Source.infer_types`), one design flaw found and corrected (ground truth keyed by
value rather than ID), ~1,400 lines of server-shaped testkit deleted, and the
ER-evaluation capability preserved and now actually asserted.

## Phase B — demolition & repackaging ✅ done

**Deleted.** `matchbox/server/**` (10,342 lines), `client/_handler/**`, the CLI,
`test/server/**`, `test/e2e/**`, `test/scripts/**`, the server test fixtures, and the
`server` optional-dependency extra. Source tree 21,445 → ~11,000 lines; tests 19,219 →
~8,700.

**Renamed and restructured.** `matchbox` → `matchlab`; `matchbox.client.*` promoted to
top level; `matchbox.common` → `matchlab.core`. Tests follow: `test/common` →
`test/core`, methodology tests → `test/methodologies`, `test/client` gone.

**DTO split done.** `dtos.py` 781 → 456 lines: auth/identity (`User`, `LoginResponse`,
`Group`, `PermissionType`), HTTP envelopes (`OKMessage`, `ErrorResponse`,
`ResourceOperationStatus`), uploads, server addressing (`StepPath`, `Run`,
`Collection`) and `Match` all deleted; the config models that *are* the serialisable
plan survive.

**Dependencies pruned.** Dropped `httpx`, `psycopg`, `cryptography`, `email-validator`,
`tenacity`, `click`, `typer`, `textual` from runtime, and `docker`, `moto`, `respx`,
`boto3-stubs`, `pytest-asyncio` from dev. Docker is warehouse-only; `environments/`
deleted; `justfile` and `test/justfile` simplified.

**Tests ported, not dropped.** The 25 methodology tests (the actual matching
algorithms) and `ResolverMatches` moved onto the plan API — `@patch.object(Query,
"data")` → `@patch.object(Clean, "_frame")`, `model.run()` → `model.collect().edges()`,
with testkit data written to the warehouse so only the cleaned frame is mocked and the
rest executes for real. Porting `ResolverMatches` caught a **third Phase A regression**:
`Source.qualify_field` had been dropped in the rewrite and nothing else exercised
`view_cluster`.

**Suite: 281 passing, 1 skipped, lint clean.** Source is now 9,228 lines (from 21,445); tests 6,796 (from 19,219).

**Naming cleanup (the rename made real, not cosmetic).**
* `_settings.py` **deleted**. `ClientSettings` was misnamed *and* entirely dead: nothing
  imported `settings`, and every field (`api_root`, `jwt`, `user`, `retry_delay`,
  `timeout`) was HTTP/auth config. `batch_size` died with `run_and_sync`.
* `exceptions.py` **586 → 60 lines**. Of 40 classes only 8 were used; the rest were HTTP
  status carriers plus a registry that reflected over them to build a wire-format enum.
  Naming now follows Polars: one prefixed base (`MatchlabError`) so `except MatchlabError`
  reads well when imported bare, with unprefixed specifics beneath it — `StepNotFound`,
  `SchemaMismatch`, `ExtractTransformError`, `SourceClientError`, `SourceTableError`,
  `NameValidationError`, `DataTypeError`. Specifics avoid shadowing builtins.
* `core/dtos.py` → `core/config.py`, **781 → 269 lines**. It holds no transfer objects
  any more — it is the serialisable description of a plan's steps. Dropped the last
  vestigial models (`Step`, `StepType`, `QueryConfig`, `ResolverConfig`, `ModelConfig`,
  `JsonObject`), all unused; `MatchboxName` → `Name`, `validate_matchbox_name` →
  `validate_name`. (The config `Step` also clashed conceptually with the plan node
  `Step`.)
* Prose, logger names and the logging entry-point group swept: **zero `matchbox`
  references remain in the Python source.**

**Outstanding in Phase B:**
* ~~`test/warehouse/test_locations.py` is skipped at module level.~~ **Fixed
  2026-07-24.** Rebuilt in `test/warehouse/conftest.py`, and it needs no container:
  `validate_extract_transform` only reads `engine.dialect.name`, and SQLAlchemy builds
  an engine without connecting, so the dialect-comparison tests get a real
  Postgres-dialect `Engine` that never opens a socket (`psycopg` added as a dev dep for
  the driver). The two SQLite clients share one temp file rather than separate
  `:memory:` databases, so a test can write through SQLAlchemy and read through ADBC.
  24 tests, no Docker; the now-unused `docker` marker and the `just test warehouse`
  recipe went with them.
* The CLI was deleted outright rather than shrunk: every command (including the eval
  TUI) started by loading a saved DAG **by name from the server**, and plan
  serialisation does not exist yet. Rebuilding it is a feature that depends on plan
  save/load, not a port. The eval *library* API is unaffected.

## Phase C — docs & release ✅ (docs done; release outstanding)

**Deleted:** `docs/server/`, `docs/client/`, `docs/api/server/`, `docs/api/client/`,
`docs/api/common/`, and the pre-0.10 migration guide.

**Written:**
* `docs/index.md` — landing page: a code sample, what it does, and an explicit *what it
  doesn't do* (no server, no accounts, nothing to deploy).
* `docs/guide/install.md`, `build-a-plan.md`, `query.md`, `evaluate.md` — the guide,
  organised around the four things you actually do. `build-a-plan.md` carries the verb
  table, the `clean(None)` vs `clean({})` distinction, layering through a resolver, and
  collect/caching semantics.
* `docs/migration/matchbox-to-matchlab.md` — a hard-break guide: why, import table,
  DAG-to-plan tabs, verb renames, removed-without-replacement, the exception rename
  table, and the five behavioural changes that bite (lazy collect, content caching,
  fresh `Source` to refresh, materialised resolution, local reclaimable storage).
* `docs/api/**` — mkdocstrings stubs against `matchlab.*`, split plan / storage /
  evaluation / core.
* `docs/use-cases.md`, `README.md`, `docs/contributing.md` rewritten.

**Swept:** `mkdocs.yml` (nav, `repo_url`, `preload_modules`), CI (deleted the JWT keygen
and `.env` steps that referenced files removed in Phase B; docs now build `--strict`),
CodeQL comment, PR template, `.vscode/launch.json` (dropped the uvicorn server profile),
`extra.css`, `personal-data-exclusions.txt`. Dropped `pydantic-settings` and
`pytest-env` — both dead once settings and `.env` went.

**Outstanding:**
* `docs/assets/matchbox-*.{svg,png}` are the old logo artwork, now unreferenced.
  `mkdocs.yml` has no `logo`/`favicon` until replacements exist — renaming the files
  wouldn't change the artwork, so this is a design decision, not a sweep.
* Cut the first `matchlab` release: create the `uktrade/matchlab` repo (or rename), claim
  the PyPI name, publish a GitHub release. `cd.yml` is already name-agnostic.

---

## Phase D — interface simplification ✅ (2026-07-24)

Not planned up front. It came out of reading the finished code and asking, step by
step, what each thing was still buying. Almost everything removed here was a shape
inherited from the server that the local library had stopped needing.

**The rule that kept recurring:** one declaration, not two. Wherever the codebase said
the same thing twice — a field list beside the SQL that selects it, a type beside the
warehouse's own, a name-validation pattern beside the identifier rules it was meant to
protect — the two could drift, and the drift was where the bugs were.

### `Source` is its query plus a key

`Source(location, name, extract_transform, key_field)`. Identity is **every column the
extract returns except the key**, so a column that shouldn't affect identity is one you
shouldn't select, and a type you want pinned is a `cast` in the SELECT.

Deleted: `index_fields`, `SourceField`, `infer_types`, `Location.infer_types`,
`schema_overrides` plumbing. `DataTypes` survives for the testkit's data generator,
which is a separate concern. Keys are cast to string on read rather than validated.
`index_fields` remains as a *property* read from the extract, so it cannot drift.

This closed a live bug found the same morning: `_config_key` hashed only the index, so
changing a selected-but-unindexed column left the fingerprint unmoved — the source
cache-hit, never re-stored, and every downstream view read the stale value. With
identity as the extract, that class is structurally impossible.

### `Cleaner` became `View`, and `combine_type` became `group`

The node was named after its optional clause. Its real job is to say **which records a
model matches over**: sources read directly are grouped by their leaves, sources read
through a resolver by that resolver's clusters. Cleaning is a projection on top. That
is why `identifiers()` looked unrelated to cleaning — it is the essential part.

`combine_type` had no users anywhere, and none of its three settings solved the problem
it existed for. `concat` left duplicate rows per entity; `set_agg` collapsed them but
wrapped every column in a list; `explode` *looked* like it produced cross-source
combinations but Polars explodes columns element-wise, so it round-tripped to its input.
A broken feature, not a redundant one.

`group=True` runs the cleaning as `GROUP BY id`, so each entity is one row and each
column says how it combines. It requires cleaning expressions — there is no defensible
default for collapsing a column. The old design could not have done this:
`Query.run(return_leaf_id=True)` refused any combine type but `concat`, because
collapsing rows destroyed the row-to-leaf mapping. Here leaves travel through
`identifiers()`, read from the adapter, never through the view frame.

### Configs are symmetric, and typed

Phase B deleted `QueryConfig`/`ModelConfig`/`ResolverConfig` as unused, leaving one
typed config and three anonymous dict literals inside `_config_key`. Restored as
`ViewConfig`, `ModelConfig`, `ResolverConfig`; every step has a `config` property and
`Step._config_key` is concrete over it, with `Source` overriding only to append its
data hash.

The invariant now lives somewhere reviewable: **a config carries everything its step's
output depends on, and nothing else.** That rule explains the one asymmetry —
`SourceConfig` records its own name because a source's name prefixes its columns and
tags its rows, while no other step's name reaches its own output. Where a name is
load-bearing to a consumer it lives in the *consumer's* config, which is why
`ResolverConfig.inputs` stays: thresholds are keyed by model name. (Superseded — those
thresholds key by input position now, and `ResolverConfig.inputs` is gone with them.)

`SourceConfig` also lost six methods that predated the split — `parents` and
`dependencies` had zero callers, and `prefix`/`qualify_field`/`f` never touched `self`
(they took the source's name as an argument, because on the server the config didn't
know it). They are `Source` methods now.

### Names do less, and are checked where it matters

Deleted `validate_name`, `Name`, and the four `*StepName` aliases — all
`TypeAlias = Name`, so no type checker could tell them apart, and the union had no
consumers. The pattern they enforced allowed `.` and `-`, which are exactly what breaks
a column identifier, and rejected characters that were harmless.

Replaced with a check on `Source.__init__` only, since only source names become
identifiers: `^[a-zA-Z_][a-zA-Z0-9_]*$`, with an error naming the column it would have
produced. Reserved words need no handling — the name is only ever a prefix.

Also deleted `Step.get_step` / `lineage.find` (no callers; hold a variable) and the
write-only `description` field on every step.

#### Abandoned (2026-07-25 → 26): making derived names carry uniqueness

Recorded because it is a tempting dead end. Porting a real pipeline hit a derived-name
clash, and the first two attempts to fix it both kept the premise that a step should be
given an invented, human-readable name:

1. **Split names by kind** — views got `Step.named = False` so only "handles" (sources,
   models, resolvers) were checked for uniqueness. Right instinct, wrong seam: what
   decides is not the *kind* of step, it is whether the author named it.
2. **Make derivation injective** — `"_".join` over source names was ambiguous (source
   `a_b` and the pair `(a, b)` both gave `a_b`), so the parts joined on separators a
   source name cannot contain (`+`, `@`, `~`), a model's stem came from its views rather
   than their sources, and `shorten_stem` bounded the compounding with a digest, because
   four sources linked pairwise derived a name several hundred characters long.

All of it was machinery to make a *generated* string unique enough to be trusted — and
a generated string is never worth trusting, because it cannot encode settings. Two
models differing only in `model_settings` derive the same name no matter how clever the
scheme. Deleted in favour of positions; see below.

#### Configs describe settings; documents describe plans (2026-07-25)

`ModelConfig.left`/`right` existed for serialisation, and were simultaneously a
spurious miss in the fingerprint. That contradiction was not a bug in either — it was
one model being asked to do two incompatible jobs:

* **identity**, hashed into a fingerprint, which must carry everything affecting the
  step's output and *nothing else* — edges included, because `_fingerprint` already
  folds in the parents';
* **description**, which must carry enough to rebuild the step, edges very much
  included.

Split, rather than compromised. `matchlab.document` holds the second job:

```
PlanDocument.steps: (StepNode, ...)   # lineage.walk order
StepNode: kind, config, inputs: (int, ...), name: str | None
```

Nodes refer to each other by **position**, which settles a question the naming work
had left open: a document that referenced steps by name would make every name
load-bearing for serialisation, and stable naming would become a hard requirement.
Positions are legitimate here because the document is a *derived view* — humans write
Python, and this is a dump target for executing a plan in another environment — so
nothing needs to be hand-authored. Positions also preserve structural sharing, where
nesting each step's inputs inside it would inline a shared view once per consumer.

Consequences:

* Configs are now settings-only, so the spurious-miss column empties. `ViewConfig`
  lost `sources` and `resolver` (no runtime consumers at all — they fed the
  fingerprint and described edges), `ModelConfig` lost `left`/`right`. `ResolverConfig`
  lost `inputs` a step later, when thresholds moved to input positions — after which
  the column is empty.
* `StepNode.name` records the name the *author published under*, null otherwise —
  there being nothing else to record, since a step is otherwise referred to by its
  position and its position is its index in `steps`. (Dropped entirely a step later,
  when publishing became an operation on the result rather than part of the plan. What
  survived is the reason: the document and the rest of the system share one notion of
  reference.)
* A document carries no client, no credentials and no data. Source content hashes are
  derived on load from the target warehouse, so same rows means same fingerprints and
  the target store hits cache; different rows means it re-runs. Both are tested.
* It cannot carry code: `model_class` is a registry name, so the target environment
  needs the same classes registered. Portable across environments, not codebases.

The acceptance criterion is the round trip, in `test/plan/test_document.py`: dump →
JSON → load → collect, and every step fingerprints identically, with a sabotaged
`_execute` proving nothing re-ran. The flagship four-source example dumps to 24 nodes
and 5.5 KB of JSON, rebuilds from the JSON alone, and hits cache on every step.

#### Names are publications, and references are objects (2026-07-26)

The end of the thread. Two changes retire the last of the naming machinery.

**Settings point at inputs by object, serialised as position.** `thresholds={model:
0.9}` — you already hold the model, so there is nothing to retype and a typo is a
`NameError` rather than a failure at collect time. `Resolver._positions` translates to
an index at construction, the only place that knows both the models and their order.
Positions rather than names because a name is not identity: `ResolverConfig.inputs`
existed solely to make a rename move the resolver's fingerprint, and with it gone
**no config anywhere records a name but its own** (`SourceConfig`, which is semantic).
The spurious-miss column is empty.

The methodology layer keys by position too, so `resolvers/components.py` imports no
plan object — which also avoids the import cycle `models.py` already dodges lazily.
Reordering inputs reassigns thresholds, but reordering already changes the fingerprint
(parents are folded in order), so that is a different resolver, not an inconsistent one.

**Steps have no names; publishing is an operation.** `Resolver.publish(name)` points a
name at a collected resolution, and that is the only name in the system apart from a
source's. See "Publishing is an operation, not a property" below for why it ended up
that way; the steps here are what got it there.

Positions were already the reference type for `PlanDocument`, so this is one answer
everywhere rather than three for three contexts: `step 7` in a run's log is `[7]` in
that plan's `draw()` and `steps[7]` in its document. Unlike a fingerprint — the other
candidate — a position exists before anything is collected, which matters because
`draw()` on a plan you are still building is when you most want to read it, and a
source cannot be fingerprinted without first reading its warehouse.

**A position is not stored on the step.** It belongs to the walk it came from — the
same node numbers differently in `walk(deduped)` and `walk(companies)` — so writing one
back would make it a lie as soon as anything walked from elsewhere. Whoever walks passes
it to whoever needs it: `collect` hands it to `_ensure`, `draw` keeps its own mapping
from `lineage.number`, which is a pure function. That is also why all per-step run
logging lives in `_ensure` rather than in `_execute`: `_ensure` is the level that has
the number, and it keeps `_execute` to the work.

A log line is `[step 2] Ran in 0.041s`, and `draw()` numbers the same walk, so the
drawing is the key to the output:

```
○ [4] resolver 'entities'
    ├── ○ [3] model
    │   └── ○ [1] clean
    │       └── ○ [0] source 'crn'
    └── ○ [2] model
        └── ○ [1] clean
            └── ○ [0] source 'crn'
```

An earlier cut had each step describe itself in prose as it ran (`NaiveDeduper deduping
crn`). Dropped: the plan says the same thing structurally, and a per-step sentence is a
second, drifting description of what the code already states. Whether and how `collect`
surfaces the plan itself is being worked out separately; it does not print one today.

That reverses the default. Comparing two methodologies over one view, or cleaning a
source several ways, now needs no names at all; before, the plan refused to collect
until you invented one for work you never meant to publish.

`lineage.validate` shrank with it — the config diff it used to produce only made sense
when neither colliding name had been chosen — and then went entirely, one section down,
when names left the plan.

*Caveats, both accepted:* a plan and a sub-plan of it number differently, so a run and
that run's drawing agree but two different roots need not. And an unnamed step is not
findable by name in a store, so a workflow that reads a resolution back must name that
resolver.

#### Publishing is an operation, not a property (2026-07-26)

`Source(name=...)` and `Resolver(name=...)` were unrelated operations wearing the same
keyword. A source's name is semantic — it prefixes every column that source contributes
and tags its rows in a resolution, so it changes the output. A resolver's was a storage
handle that touched nothing about the plan. That asymmetry was the tell.

The two are now different words as well as different mechanisms. A **name** belongs to
a source and is part of its output; a **label** belongs to the store and points at a
resolution. `name` is therefore unambiguous everywhere it still appears.

A label does not affect the output, the fingerprint, or the plan, so it has no business
in the plan's definition. It became an act on the result instead:

```python
entities = crn_dedupe.resolve(dh_dedupe).collect().publish("entities")
```

Which killed rather more than the keyword:

* **`lineage.validate` is gone.** Its only remaining job was the published-label check.
  Cycles cannot be constructed, so with names gone it had nothing left to do; `collect`
  no longer calls it, and `_clash_message` went with it.
* **`Step.name` and `given_name` are gone.** `Source.name` is the source's own
  attribute, stated in the type rather than papered over by a shared base.
* **`store_model(fp, name)` / `store_resolver(fp, name, …)` lost their name argument,**
  and `artifacts` lost its `name` column. Writing an artifact and labelling one were
  always two acts; now they are two methods.
* **`PlanDocument` sheds `StepNode.name`.** Publishing is done to a result, so it is
  not part of the plan a document describes. The only names left in a document are
  sources', in the only sense the word still has.

The alias table arrived as part of it rather than as separate work: `labels(label PK,
fp, published_at)`, with `publish` as an explicit `INSERT OR REPLACE`. That fixes what
`find` used to be — a `fetchone` over however many generations shared a name, returning
an arbitrary one — and moves the overwrite decision to the caller, where
`Resolver.publish` raises unless `overwrite=True`. Republishing a label for the same
fingerprint is a no-op, so re-running an unchanged pipeline does not fail on the second
run. Purging an artifact drops any label pointing at it, so a label never resolves to
something that is gone.

The check also got *more* correct rather than merely relocated: in-plan uniqueness was
a proxy for the thing that actually collides, which is labels in a store, across runs.
`publish` checks the real namespace.

*Costs, both accepted:* publishing is now forgettable — a script that used to declare
`name=` needs a call you can omit, and you find out when something cannot locate your
resolution later (mitigated because a miss lists the known labels). And a transferred
plan no longer carries an intended label, so the receiving environment publishes under
whatever it wants.

### Loose ends closed

* **`test/warehouse/test_locations.py` un-skipped.** It never needed Docker:
  `validate_extract_transform` only reads `engine.dialect.name`, and SQLAlchemy builds
  an engine without connecting. The real blocker was that the two SQLite fixtures were
  separate `:memory:` databases, so a test wrote through SQLAlchemy and read through
  ADBC. They now share a temp file. The `docker` marker and the `just test warehouse`
  recipe went with them.
* **Suite: 324 passing, 0 skipped**, with no container required for any of it.

### The eval TUI is back, as a library call

`matchlab.eval.review(resolver, tag=..., sample_file=...)`. It turned out the app was
barely server-coupled: `EvaluationItem` is unchanged, so `widgets/` and `modals.py`
(~21 KB, the bulk of the UI) ported verbatim, and `app.py` needed three edits —
`_handler.send_eval_judgement` → `adapter.store_judgement`, a `Resolver` object in
place of a `DAG` + resolver path, and one dead import.

Everything server-shaped lived in the CLI wrapper (`--collection`,
`DAG(...).load_default()`, `--warehouse` to reattach clients, `--resolver` by name),
which is exactly the part that needs plan serialisation. Replaced by a function that
takes the live object you already built, so nothing is blocked.

A sample file is *not* plan-free, contrary to first appearances: it records which
records were shown together, not their values, so `resolver.sources` is read either
way to display them. `sample_file` only bypasses `adapter.sample()`.

Textual is an optional extra (`matchlab[tui]`); `pytest-asyncio` in `auto` mode runs
Textual's `run_test()` pilot. 11 tests, over a real plan and a real DuckDB store.

**And a command again.** `matchlab review pipeline:entities` — argparse, no new
dependency, registered as `[project.scripts]`. The target is `module:attribute`, as
uvicorn and celery do it, which sidesteps serialisation entirely: Python builds the
plan, so the clients come attached. `matchlab version` is the only other command; the
old CLI's `health`/`auth`/`collections`/`groups`/`admin` were all server operations.

There is deliberately no `matchlab run`: a pipeline is a Python file, and
`python pipeline.py` runs it.

**And the store is now self-describing**, which is what makes
`matchlab review entities --store run.duckdb` work with no plan and no warehouse:

* `get_samples` reads record values from the **stored extract** rather than re-fetching.
  That is also the more correct thing to judge — the data the matching saw, not what
  the warehouse says today.
* Models and resolvers record their `name` (only sources did), so a store can be
  browsed: `adapter.names("resolver")`, `adapter.find(kind, name)`.
* A source records its `key_field`, so its extract can be joined to a resolution
  without the plan.
* A resolution records the **source fingerprints** it covers. It names its sources, and
  one store can hold several generations of a name, so the names alone were ambiguous.
* `meta.schema_version` guards all of this: a store from an older matchlab is recreated
  rather than half-read. Artifacts are a cache, so nothing is lost.

This is the same goal as plan serialisation, reached from the other end — the store
already held everything, it just wasn't labelled. What it does *not* give you is a plan
you can re-run; for that, the edges and client-reattachment problems remain.

### Still open

* **TODO(fingerprints)** — see Known limitations below. Unchanged by Phase D, except
  that the observable-interface hazard now has a name: `View.identifiers()`.
* **The CLI** — still waiting on plan serialisation. The typed configs from Phase D are
  most of what a serialiser needs; what's missing is edges (steps refer to each other by
  name, and a name is not a node) and a decision about how location clients are
  reattached on load. **The eval TUI no longer waits on it** — see below.
* **`Source.dedupe`/`link` and the testkit `view()` wrappers** now have honest
  signatures, but `SourceTestkit`/`ResolverTestkit` still forward `*args, **kwargs`.

---

## Design decisions (settled)

### Record identity is a hash of content, not the key

A leaf ID — a record's identity — is a hash of its **content** (every non-key column),
computed in `Source._read_warehouse` and turned into an ID by
`core.resolution.leaf_id`. Not the key. Questioned 2026-07-24 (the server hashed to save
storage, which is gone locally — so is the hashing still needed?), and re-affirmed for a
reason independent of storage.

The reason is evaluation. A judgement is a person or model saying *"looking at this,
these records are the same entity."* The only input to that decision is the content
shown — a name, a postcode. The key plays no part; the judge never sees it. So a
judgement must be anchored to the content it was made against:

* **Key-anchored**, a judgement outlives a content change. A match decided on "acme /
  london" still stands after the row becomes "acme / manchester" — a decision credited
  to evidence the judge never saw. It reads as inexplicable later, or silently ratifies
  a match nobody made.
* **Content-anchored**, the leaf ID changes when the content does, so the old judgement
  stops applying. Nobody judged the new content; there should be no judgement about it.
  Judgements decay exactly when their evidence does.

The key's *stability under content change* — which first looked like an argument for
using it — is precisely why it's the wrong anchor. Same logic settles index-time
"dedup": identical rows share a leaf because to a judge they are indistinguishable;
there is no decision to make, and the differing keys don't separate them because a key
is not evidence.

**Invariant it rests on: what is hashed must equal what is shown.** Hash a column the
sampler doesn't display and a judgement decays for a reason the judge can't see; display
one that isn't hashed and a judgement survives a change the judge can see. They line up
today — `leaf_id` covers every non-key column, `get_samples` shows every non-key column
— and "the extract is the whole declaration" (a selected column is both hashed and
shown) is what keeps them lined up. A future change that hashes or displays a strict
subset of the other breaks evaluation subtly, not loudly.

Consequence, not cause: content-addressing also makes runs reproducible and leaf IDs
stable across re-collect. Real, but downstream of the above — don't cite it as the
reason.

---

## Known limitations

### Fingerprints are input-addressed, not output-addressed

`Step._fingerprint()` is `H(kind, config, parents' fingerprints)` — derived from the
**plan**, not from what the step produces. This is deliberate and load-bearing: the key
has to exist *before* the step runs, because that is the only way `_ensure` can call
`adapter.has(fp)` and skip the work. An output digest is only knowable once the work is
already done, which is too late to avoid it.

Sources are the deliberate exception. `Source._config_key()` folds
`hash_arrow_table(hashes)` — a hash of the data it read — into the key, because no
amount of configuration reveals that the warehouse moved underneath you. Data enters the
plan at sources, so that is the one place content-addressing belongs. (Not to be
confused with `core/resolution.py`'s `leaf_id`/`root_id`, which *are* content-addressed:
those are record and cluster identity inside the data, a different hash for a different
job.)

The cost is that the key can disagree with the bytes, in both directions:

* **Stale hit** — config omits something that changes the output, so the step is never
  re-run and the old artifact is read instead. This is the dangerous direction, and the
  only defence is the discipline that every `_config_key` covers everything its step's
  output depends on. Live instance: `SplinkLinker` training functions that sample
  (`estimate_u_using_random_sampling`) are non-deterministic unless given a `seed` in
  `arguments`. Same settings, same fingerprint, different edges — first write wins,
  silently. Documented on `SplinkSettings`.
* **Spurious miss** — config includes something that *doesn't* change the output, so
  work is redone for nothing. **There is currently no instance of this.** Every field
  in every config either changes the step's output or is the step's own semantic name
  (`SourceConfig.name`, which prefixes its columns). No config records an input's name,
  and settings that point at an input use its position, which does change the output —
  it decides which model a threshold applies to. Worth protecting: the way this
  reappears is somebody recording a *description of an input* rather than a setting.
* **No early cutoff** — a `View` whose SQL is reformatted but semantically unchanged
  invalidates everything below it, because the cascade is keyed on config all the way
  down rather than stopping where the data stops changing.

`_purge` keeps the stored artifacts themselves consistent, and is necessary rather than
cosmetic: `source_leaves`, `model_edges` and `resolution` are shared tables written with
plain `INSERT`, so without it a second store would duplicate rows rather than replace
them.

**Fixed once already (2026-07-24).** `Source._config_key` hashed only `hashes`, which
content-addresses rows by their *index fields*. Any other column the extract/transform
selected — indexed on company and postcode, town riding along for a cleaning expression
— was invisible to the fingerprint, so changing it in the warehouse produced a cache hit,
the source never re-stored, and every downstream view kept reading the stale column. The
extract is now hashed too. Regression test:
`test_a_change_to_a_non_indexed_column_invalidates_the_source`.

### TODO(fingerprints): action key over content-addressed store

Split the single key in two, as Bazel does with its action cache over the CAS, or Nix
with content-addressed derivations:

```
action key (plan-derived, as now)  →  output digest (content-derived)  →  artifact
```

Steps still *look up* by action key, but a child's key is built from its parents'
**output digests** instead of their action keys. Change something that doesn't change the
output and the parent gets a new action key, runs, produces the same digest, and every
child hits cache.

The honest ledger — an output digest governs **propagation**, never **admission**:

| | |
|---|---|
| Early cutoff | **gained** — the actual benefit |
| Storage dedup for equivalent artifacts | gained |
| Rename tolerance (re-runs the step, spares the subtree) | gained |
| Stale hits from dishonest config keys | **unchanged** — a step whose action key hits still never runs, so its digest is never consulted. Seeding Splink stays mandatory. |
| Work-set knowable before running | **lost** — past the first miss a child's key depends on output that doesn't exist yet, foreclosing a future `.explain()` |

Implementation is smaller than it sounds. Redefine `_fp` to mean *artifact address =
output digest* and add `_action_key()` alongside; `_fingerprint`'s existing
`parts.append(parent._fp)` then already reads "build my key from my parents' digests",
and every `read_*(step._fp)` site keeps working because artifacts move to living at the
digest. Of the ~22 `_fp` references in `src/`, about three change. `hash_arrow_table` is
already invariant to row and field order, which is the precondition.

Two hazards:

* `hash_arrow_table` returns the constant `b"empty_table_hash"` for any empty table, so
  the digest preimage **must** include the kind or every empty artifact collides at one
  address — the Phase 2 bug returning in a worse form.
* Early cutoff assumes a step's digest determines everything its descendants observe.
  `View.identifiers()` breaks that: it reads `source.name` off the live step object and
  that name lands in the resolver's `source` column, while `View._compute` drops it
  before returning. Rename a source in an explicitly-named plan and the resolver would
  hit cache on a resolution carrying the old name. Fix is principled — a step's digest
  covers its whole *observable interface*, so `View` digests `(frame, identifiers)` —
  but every other cross-step read needs the same audit, and missing one is silent
  corruption.

**Verdict (2026-07-24): not now.** With the stale-hit claim withdrawn, the remaining win
is early cutoff in a narrow band — config changes are mostly normalised away already by
`json.dumps(sort_keys=True)`, so it comes down to warehouse churn in columns that
cleaning projects away, and only when cleaning is an explicit projection. Not worth a new
silent-corruption surface plus the loss of a precomputable work-set until someone has
felt the re-run cost with a profile behind it.

---

## Risks

* **Phase A is a rewrite of the step layer.** The validated core (adapter, resolution
  engine, components methodology, hashing) is untouched, but every node class and all
  integration tests move. Sequenced before demolition so the interface is proven while
  the old suite still exists as a reference.
* **Cache invalidation** — "collect re-runs only uncached ancestors" must stay airtight
  across the move to identity-based lineage.
* **GC correctness** — fingerprint sharing between distinct nodes makes sweep-vs-delete
  the critical detail.
* **`common` untangling** — the DTO transport-vs-config split remains the load-bearing
  cut in Phase B.
