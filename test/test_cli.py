"""The `matchlab` command.

Only the argument handling and target loading are tested here — the reviewer itself
has its own tests, and launching a full-screen app from a unit test proves nothing.
"""

import importlib
import sys
from pathlib import Path

import pytest

from matchlab import Resolver
from matchlab.cli import _load_target, main

PIPELINE = """
from sqlalchemy import create_engine, text

from matchlab import read_database
from matchlab.models.dedupers import NaiveDeduper

_engine = create_engine("sqlite:///{db}")
with _engine.begin() as conn:
    conn.execute(text("CREATE TABLE crn (pk TEXT, company TEXT)"))
    conn.execute(text("INSERT INTO crn VALUES ('a1','acme'),('a2','acme')"))

_source = read_database(
    "crn",
    sql="select pk, company from crn",
    client=_engine,
    key_field="pk",
)
entities = _source.dedupe(
    model_class=NaiveDeduper,
    model_settings={{"unique_fields": ["crn_company"]}},
).resolve()


def build():
    return entities


not_a_resolver = "just a string"
"""

# The autouse `store` fixture (from `test/conftest.py`) keeps every test off the real
# store in the user's cache directory.


@pytest.fixture
def pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Write a pipeline module and make it importable, as a user's script would be."""
    (tmp_path / "pipeline.py").write_text(PIPELINE.format(db=tmp_path / "wh.sqlite"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    # Otherwise the second test to run imports the first test's module.
    sys.modules.pop("pipeline", None)
    importlib.invalidate_caches()
    return "pipeline"


def test_version(capsys: pytest.CaptureFixture) -> None:
    """`matchlab version` prints the installed version."""
    main(["version"])
    assert capsys.readouterr().out.strip()


def test_target_named_resolver(pipeline: str) -> None:
    """A `module:name` target resolves to that resolver."""
    assert isinstance(_load_target(f"{pipeline}:entities"), Resolver)


def test_target_factory(pipeline: str) -> None:
    """So a plan needing a live connection isn't built at import time."""
    assert isinstance(_load_target(f"{pipeline}:build"), Resolver)


def test_target_no_colon_explained(pipeline: str) -> None:
    """A target without a colon is explained, not a traceback."""
    with pytest.raises(SystemExit, match="module:attribute"):
        _load_target(pipeline)


def test_target_unimportable_explained() -> None:
    """An unimportable module is explained."""
    with pytest.raises(SystemExit, match="Could not import 'nope'"):
        _load_target("nope:entities")


def test_target_missing_attr_explained(pipeline: str) -> None:
    """A missing attribute is explained."""
    with pytest.raises(SystemExit, match="has no attribute 'absent'"):
        _load_target(f"{pipeline}:absent")


def test_target_not_resolver_explained(pipeline: str) -> None:
    """A target that isn't a resolver is explained."""
    with pytest.raises(SystemExit, match="is a str, not a Resolver"):
        _load_target(f"{pipeline}:not_a_resolver")


def test_review_launches_with_args(
    pipeline: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command's job is to turn flags into a `review()` call."""
    captured: dict = {}

    def fake_review(resolver: Resolver, **kwargs: object) -> None:
        captured["resolver"] = resolver
        captured.update(kwargs)

    monkeypatch.setattr("matchlab.eval.review", fake_review)

    store = tmp_path / "store.duckdb"
    main(
        [
            "review",
            f"{pipeline}:entities",
            "--samples",
            "3",
            "--tag",
            "session-1",
            "--seed",
            "7",
            "--store",
            str(store),
        ]
    )

    assert isinstance(captured["resolver"], Resolver)
    assert captured["n"] == 3
    assert captured["tag"] == "session-1"
    assert captured["store"] is not None
    assert captured["seed"] == 7


def test_review_several_targets(pipeline: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Naming two resolvers reviews their merged clusters: one session judges both."""
    captured: dict = {}

    def fake_review(resolver: object, **kwargs: object) -> None:
        captured["resolver"] = resolver
        captured.update(kwargs)

    monkeypatch.setattr("matchlab.eval.review", fake_review)

    main(["review", f"{pipeline}:entities", f"{pipeline}:entities"])

    assert isinstance(captured["resolver"], list)
    assert len(captured["resolver"]) == 2
    assert all(isinstance(one, Resolver) for one in captured["resolver"])


def test_logging_to_file(tmp_path: Path) -> None:
    """A TUI and a stream handler can't share a terminal."""
    import logging  # noqa: PLC0415

    from matchlab.cli import _redirect_logging  # noqa: PLC0415

    root = logging.getLogger()
    original_handlers, original_level = root.handlers[:], root.level
    try:
        log = tmp_path / "run.log"
        _redirect_logging(str(log))
        logging.getLogger("matchlab.test").info("hello")
        for handler in root.handlers:
            handler.flush()
        assert "hello" in log.read_text()
    finally:
        # _redirect_logging deliberately clears the root logger; put it back so the
        # rest of the session still logs.
        for handler in root.handlers[:]:
            handler.close()
            root.removeHandler(handler)
        for handler in original_handlers:
            root.addHandler(handler)
        root.setLevel(original_level)


def test_target_module_plan(pipeline: str) -> None:
    """Sanity: the target really is a live plan, clients and all."""
    resolver = _load_target(f"{pipeline}:entities")
    resolver.collect()
    lookup = resolver.get_lookup()
    assert lookup.height > 0
