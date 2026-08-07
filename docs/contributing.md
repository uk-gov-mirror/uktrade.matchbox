This document describes how to get started developing matchlab.

## Dependencies

* [Python 3.11+](https://www.python.org)
* [uv](https://docs.astral.sh/uv/)
* [just](https://just.systems/man/en/)
* [Docker](https://www.docker.com) — for the TruffleHog secret scan, which pre-commit
  runs on every commit.

## Setup

This project is managed by [uv](https://docs.astral.sh/uv/), linted and formatted with
[ruff](https://docs.astral.sh/ruff/), type checked with [ty](https://docs.astral.sh/ty/),
and tested with [pytest](https://docs.pytest.org/en/stable/). Documentation is built
with [mkdocs](https://www.mkdocs.org).

Install all dependencies:

```shell
uv sync
```

There is no `.env` to configure. matchlab reads no environment variables: warehouse
connections are passed to `Location`, and storage is passed to `collect()` or set with
`set_default_adapter()`.

Secret scanning is done with [TruffleHog](https://github.com/trufflesecurity/trufflehog).

For security, use of [pre-commit](https://pre-commit.com) is expected. Ensure your hooks
are installed:

```shell
pre-commit install
```

We also mandate [git trailers](https://git-scm.com/docs/git-interpret-trailers) to
confirm your local hooks ran. Ensure pre-commit has the right permissions:

```shell
pre-commit install --install-hooks --overwrite -t commit-msg -t pre-commit
```

Task running is done with [just](https://just.systems/man/en/). To see all available
commands:

```shell
just -l
```

## Run tests

The whole suite runs on Python alone — DuckDB in memory for storage, SQLite in a temp
file for warehouses:

```shell
just test
```

No container is needed, including for the tests that compare SQL dialects.
`validate_extract_transform` only asks a client what dialect it speaks, and SQLAlchemy
answers that without connecting, so those tests use a Postgres-dialect engine with no
driver behind it.

If you want a real warehouse to point matchlab at while developing, bring up whatever
database you like and pass its client to a `Location`. matchlab has no opinion about
where it runs, and ships nothing to manage one.

## Documentation

```shell
just docs
```

Serves the site with live reload. The build is run in strict mode in CI, so broken
cross-references fail the build — worth checking before you push:

```shell
uv run mkdocs build --strict
```

## Releasing

We release matchlab by creating and publishing a GitHub release from `main`. Tags must
follow [semantic versioning](https://semver.org) in the form `vX.X.X` (for example,
`v1.2.3` for a patch release or `v2.0.0` for a major release with breaking changes).

Publishing the release triggers the CD workflow, which builds and publishes the Python
package to PyPI and deploys the documentation to GitHub Pages.

## Standards

### Code

When contributing to matchlab and its associated repos, we try to follow consistent
standards. Python code should be:

* Unit tested, and pass new and existing tests
* Documented via docstrings, in the [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
* Linted and auto-formatted (`just format`)
* Type hinted and checked (`just check`)
* Structured as a Python package with `pyproject.toml`
* Using dependencies managed automatically by uv
* Integrated with the justfile when relevant

### Steps

New plan steps subclass [`Step`](api/steps.md). A step must:

* Hold references to its inputs, and to nothing downstream of it
* Return a stable `_config_key()` covering everything that changes its output, so
  caching is correct
* Do all its work in `_execute()`, reading inputs from the adapter by fingerprint

### Adapters

New storage backends subclass [`Adapter`](api/adapters.md). Beyond the read and write
methods, `stats()` has to answer for the store's size and contents — every collect
reports it, and a store nobody can measure is one that fills a disk quietly.

`trim()` is the other half, and the one to be careful with, since it deletes. Three
rules it has to hold to:

* **Keep what the caller named, and never work out the rest for yourself.** 
* **Keep every published label, listed or not**, along with whatever its resolver output
  needs to stay readable — and never touch stored judgements, which are the one thing in
  a store that cannot be recomputed.
* **Report what you actually reclaimed**, measured. Deleting and reclaiming are not the
  same number in every backend.


### Git

We commit as frequently as possible. We keep our commits as atomic as possible. We never
push straight to main, instead we merge feature branches. Before merging to main,
branches are peer reviewed.

!!! warning
    Pre-commit **must** be turned on. Any secrets you commit to the repo are your own
    responsibility.

### AI

In order to help reviewers prioritise their time appropriately, we expect any use of AI
to be declared in your PR comment.

### Actions

In order to avoid supply chain attacks, we
[pin all actions in workflows](https://codeql.github.com/codeql-query-help/actions/actions-unpinned-tag/).

When upgrading actions, we expect PR comments to confirm that the new commit is safe.
You need to cover:

* That the commit's `action.yml` only uses pinned child actions, if it has children
* That there are no critical security concerns raised in the issues

We suggest using tools like [`wayneashleyberry/gh-act`](https://github.com/wayneashleyberry/gh-act)
to help manage this, allowing you to perform the upgrade in a single line:

```shell
gh act update --pin
```

You will still need to independently verify that the new pins are safe.
