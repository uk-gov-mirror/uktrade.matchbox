"""The `matchlab` command.

Only the argument handling and target loading are tested here — the reviewer itself
has its own tests, and launching a full-screen app from a unit test proves nothing.
"""

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from matchlab import Resolver, set_default_adapter
from matchlab.adapters import DuckDBAdapter
from matchlab.cli import _load_target, main

PIPELINE = """
from sqlalchemy import create_engine, text

from matchlab import Source
from matchlab.locations import RelationalDBLocation
from matchlab.models.dedupers import NaiveDeduper

_engine = create_engine("sqlite:///{db}")
with _engine.begin() as conn:
    conn.execute(text("CREATE TABLE crn (pk TEXT, company TEXT)"))
    conn.execute(text("INSERT INTO crn VALUES ('a1','acme'),('a2','acme')"))

_location = RelationalDBLocation(name="warehouse", client=_engine)
_source = Source(
    location=_location,
    name="crn",
    extract_transform="select pk, company from crn",
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


@pytest.fixture(autouse=True)
def adapter() -> Iterator[DuckDBAdapter]:
    """Never let a test touch the real store in the user's cache directory."""
    store = DuckDBAdapter(":memory:")
    set_default_adapter(store)
    yield store
    set_default_adapter(None)
    store.close()


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


def test_version_prints_the_installed_version(capsys: pytest.CaptureFixture) -> None:
    main(["version"])
    assert capsys.readouterr().out.strip()


def test_a_target_resolves_to_the_named_resolver(pipeline: str) -> None:
    assert isinstance(_load_target(f"{pipeline}:entities"), Resolver)


def test_a_target_may_name_a_factory(pipeline: str) -> None:
    """So a plan needing a live connection isn't built at import time."""
    assert isinstance(_load_target(f"{pipeline}:build"), Resolver)


def test_a_target_without_a_colon_is_explained(pipeline: str) -> None:
    with pytest.raises(SystemExit, match="module:attribute"):
        _load_target(pipeline)


def test_an_unimportable_module_is_explained() -> None:
    with pytest.raises(SystemExit, match="Could not import 'nope'"):
        _load_target("nope:entities")


def test_a_missing_attribute_is_explained(pipeline: str) -> None:
    with pytest.raises(SystemExit, match="has no attribute 'absent'"):
        _load_target(f"{pipeline}:absent")


def test_a_target_that_is_not_a_resolver_is_explained(pipeline: str) -> None:
    with pytest.raises(SystemExit, match="is a str, not a Resolver"):
        _load_target(f"{pipeline}:not_a_resolver")


def test_review_is_launched_with_the_parsed_arguments(
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
            "--store",
            str(store),
        ]
    )

    assert isinstance(captured["resolver"], Resolver)
    assert captured["n"] == 3
    assert captured["tag"] == "session-1"
    assert captured["adapter"] is not None
    assert captured["sample_file"] is None


def test_logging_is_redirected_to_a_file(tmp_path: Path) -> None:
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


def test_a_plan_built_in_the_module_is_usable(pipeline: str) -> None:
    """Sanity: the target really is a live plan, clients and all."""
    resolver = _load_target(f"{pipeline}:entities")
    resolver.collect()
    lookup = resolver.get_matches().as_lookup()
    assert lookup.height > 0
