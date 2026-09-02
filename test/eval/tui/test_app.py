"""The review app, driven end to end over a real plan.

There is no server and no mocked handler. Samples come from a collected resolver, and
judgements land in the same DuckDB store the rest of the library uses. That round trip
(sample, paint, store, score) is the thing worth testing.
"""

from collections.abc import Callable
from pathlib import Path

import pytest
from sqlalchemy import Engine

from matchlab import Resolver, Source
from matchlab.eval import EvalData
from matchlab.eval.tui.app import EntityResolutionApp
from matchlab.models.dedupers import NaiveDeduper
from matchlab.stores import DuckDBStore

# `warehouse`, `store` and `source` come from `test/conftest.py`. Only `crn` is read.


@pytest.fixture
def resolver(source: Callable[..., Source]) -> Resolver:
    """Crn deduplicated on `company`, for the review app to sample from."""
    crn = source("crn")
    return crn.dedupe(
        model_class=NaiveDeduper,
        model_settings={"unique_fields": [crn.f("company")]},
    ).resolve()


def _app(resolver: Resolver, **kwargs: object) -> EntityResolutionApp:
    # Debouncing is a UI nicety that only makes tests wait.
    return EntityResolutionApp(resolver=resolver, scroll_debounce_delay=None, **kwargs)


async def test_app_loads_clusters(
    resolver: Resolver,
) -> None:
    """The app opens with the collected resolver's clusters queued."""
    resolver.collect()

    app = _app(resolver, num_samples=5)
    async with app.run_test():
        assert app.queue.total_count > 0
        assert app.current_item is not None
        # The records on screen are the ones that were grouped.
        assert set(app.current_item.records.columns) >= {"leaf"}


async def test_app_stores_judgement(resolver: Resolver, store: DuckDBStore) -> None:
    """Paint every group, submit, and the judgement is stored and scoreable."""
    resolver.collect()

    app = _app(resolver, num_samples=5, session_tag="review-test")
    async with app.run_test() as pilot:
        assert app.current_item is not None

        # Judge until a cluster with more than one record has been submitted. A
        # singleton yields no pairs to score, and which cluster leads is sampling
        # order. This test should not depend on that order. The queue refills, so
        # bound the loop rather than draining it.
        for _ in range(10):
            session = app.queue.current
            if session is None:
                break
            groups = session.item.get_unique_record_groups()
            session.assignments = dict.fromkeys(range(len(groups)), "a")
            await app.action_submit()
            await pilot.pause()
            if len(groups) > 1:
                break
        else:  # the fixture always has a multi-record cluster
            pytest.fail("no multi-record cluster was offered")

    judgements, expansion = store.read_eval_data(tag="review-test")
    assert judgements.height > 0
    assert expansion.height > 0

    # It also scores judged pairs against the resolver's output.
    precision, recall = EvalData(store, tag="review-test").precision_recall(resolver)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0


async def test_app_seeded_session(resolver: Resolver) -> None:
    """Reviewing the clusters someone else was shown: same store, same seed.

    This is what replaced handing round a dumped sample file. The seed has to move on
    between refills or the queue would refetch the same clusters and starve on its own
    dedupe, so what is reproducible is the session, starting with its first cluster.
    """
    resolver.collect()

    first: list[list[int]] = []
    for _ in range(2):
        app = _app(resolver, num_samples=1, seed=7)
        async with app.run_test():
            assert app.current_item is not None
            first.append(sorted(app.current_item.leaves))

    assert first[0] == first[1]


async def test_app_no_samples(
    resolver: Resolver, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No samples to review is handled, not crashed on."""
    resolver.collect()
    monkeypatch.setattr(
        "matchlab.eval.tui.app.get_samples", lambda **_kwargs: {}, raising=True
    )

    app = _app(resolver)
    async with app.run_test():
        assert app.queue.total_count == 0


async def test_app_reviews_without_plan(
    warehouse: Engine, source: Callable[..., Source], tmp_path: Path
) -> None:
    """The reviewer needs neither the plan nor the warehouse.

    That is the point of storing extracts. Collect into a file-backed store, throw the
    plan away, and dispose of the warehouse engine. The reviewer still shows real
    values. They came from the extract cached at collect time, which is the data the
    matching actually saw.
    """
    store = DuckDBStore(tmp_path / "run.duckdb")
    crn = source("crn")
    plan = crn.dedupe(
        model_class=NaiveDeduper,
        model_settings={"unique_fields": [crn.f("company")]},
    ).resolve()
    plan.collect(store).publish("entities")

    del plan, crn

    # Not just "don't use the warehouse". Make it impossible to.
    warehouse.dispose()
    (tmp_path / "wh.sqlite").unlink()

    assert store.labels() == ["entities"]

    app = EntityResolutionApp(
        resolver="entities", store=store, scroll_debounce_delay=None
    )
    async with app.run_test():
        assert app.current_item is not None
        # Real values, not just identities.
        columns = set(app.current_item.records.columns)
        assert {"crn_company", "crn_town"} <= columns
    store.close()


def test_app_resolver_or_label(resolver: Resolver, store: DuckDBStore) -> None:
    """One parameter, two ways of saying which resolver, and they agree.

    The object and the label differ only in how the fingerprint is found, so nothing
    downstream needs to know which was given.
    """
    from matchlab.eval import get_samples  # noqa: PLC0415

    resolver.collect(store).publish("entities")

    # Sampling is random per call, so compare the whole population rather than a draw.
    by_object = get_samples(n=999, resolver=resolver, store=store)
    by_label = get_samples(n=999, resolver="entities", store=store)
    assert by_object and set(by_object) == set(by_label)


async def test_app_unknown_label(resolver: Resolver, store: DuckDBStore) -> None:
    """An unknown label names the labels that do exist."""
    from matchlab.core.exceptions import SourceTableError  # noqa: PLC0415
    from matchlab.eval import get_samples  # noqa: PLC0415

    resolver.collect()
    with pytest.raises(SourceTableError, match="under the label 'nope'"):
        get_samples(n=1, resolver="nope", store=store)
