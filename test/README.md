# Testing matchlab

These tests measure one thing: whether an entity-resolution pipeline resolves correctly. Favour a few well-chosen tests that tell a clear story over broad, shallow coverage.

## The rule for keeping a test

A test earns its place only if it can fail for a reason no other test would catch.

Before adding a test, name the one regression or invariant it guards. If the reason is "the code obviously works", or another test already covers the path, do not add it.

A test also earns its place when it guards a fixture or pins a past regression. Say so in the docstring.

## Test against real objects

Use real collaborators: a real SQLite warehouse, a real `DuckDBStore`. Mock only a boundary that is external, expensive, or non-deterministic, and name the reason in the test. The standing exceptions are the warehouse read, Splink, the TUI render surface, and the logging handlers.

A test that breaks on a refactor that preserves behaviour is testing the wrong thing. Assert on what a caller can observe, not on internal structure.

## Cover methodologies with the oracle

The `testkit` generates data with a known answer and scores a result against it. Use it for every methodology test rather than hand-building fixtures. Adding a methodology is then a new `pytest.param`, not a new test body.

It answers two questions. Pick one:

- Does the *methodology* work? Use `diff_model_edges`, with the view read mocked.
- Does the *plan* carry grouping forward? Use `diff_resolution`, with a real `collect()`.

## Name tests as tags, not descriptions

A test name categorises a test; it does not explain it. Read a file's `def test_*` lines together and they should read as an index: a shared prefix names the feature under test, and a short suffix names the case. When a file works through a set of cases, the names line up into a column you can scan.

```python
def test_match_one_to_many(...) -> None:
    """A record matches a target that already holds several IDs."""

def test_match_one_to_none(...) -> None:
    """A record matches a target that holds no IDs yet."""
```

The suffix names a case you could have listed in advance — `one_to_none`, `with_dedupe_resolver`, `rejects_bad_schema` — not the assertion. A name that reads as a sentence (`test_collapses_a_pair_scored_twice_to_its_highest`) is describing, not tagging: find its family and its case, and move the sentence to the docstring. The name having stopped explaining, the docstring now carries the reason the test exists.

Where a family's cases are a parametrised list, the case is the `pytest.param(..., id=)`, so the function name is the family alone. A lone test still takes a family prefix naming its feature — a family of one is fine, an invented one is not.

Draw both the family and the case from `docs/glossary.md`, so a name shares a vocabulary with the concept it tags.

## Where a test lives

The test tree shadows `src`. `test/<module>/test_<thing>.py` tests `src/matchlab/<module>/<thing>.py`. Cross-module, end-to-end behaviour lives in one separate area, not forced into a module.

Two fixture idioms cover most needs:

- The `crn`/`dh` scenario, a small SQLite warehouse, for plan and end-to-end tests that assert on known rows.
- The `linked_sources_factory` oracle, for methodology correctness at scale.

## Conventions

- Parametrise with `pytest.param(..., id=...)` for every case. Pass the parameter names as a tuple, with a trailing comma for a single name. Keep the signature typed.
- Give each test a docstring describing the behaviour under test.
- Prefer `unittest.mock.patch` to the `monkeypatch` fixture.
- One test, one reason to fail. Split a test that checks several behaviours.
