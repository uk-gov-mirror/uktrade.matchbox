"""Sampling and scoring across more than one resolver.

One source, two ways of deduplicating it:

    crn (pk, company, town):  a1=(acme,london) a2=(acme,leeds) a3=(beta,london)

Dedupe on `company` and you get {a1,a2}, {a3}. Dedupe on `town` and you get {a1,a3},
{a2}. The two disagree about every record, which is exactly when you want to judge them
against the same clusters.
"""

from collections.abc import Callable
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine, text

from matchlab import Resolver, Source
from matchlab.core.exceptions import SourceTableError
from matchlab.eval import EvalData, get_samples
from matchlab.eval.judgements import Judgement
from matchlab.models.dedupers import NaiveDeduper
from matchlab.stores import DuckDBStore

# The `store` fixture comes from `test/conftest.py`.


@pytest.fixture
def warehouse(tmp_path: Path) -> Engine:
    """One source, and an `a3` that agrees with `a1` on town but not company.

    Overrides the shared scenario: `a3=(beta,london)` here. Deduping on `company` gives
    {a1,a2},{a3}. Deduping on `town` gives {a1,a3},{a2}. They disagree about every
    record, exactly the case worth scoring two resolvers against one set of judgements.
    """
    engine = create_engine(f"sqlite:///{tmp_path / 'wh.sqlite'}")
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE crn (pk TEXT, company TEXT, town TEXT)"))
        conn.execute(
            text(
                "INSERT INTO crn VALUES "
                "('a1','acme','london'),('a2','acme','leeds'),('a3','beta','london')"
            )
        )
    return engine


def _dedupe_on(src: Source, field: str) -> Resolver:
    return src.dedupe(
        model_class=NaiveDeduper,
        model_settings={"unique_fields": [src.f(field)]},
    ).resolve()


@pytest.fixture
def by_company(source: Callable[..., Source]) -> Resolver:
    """Crn deduplicated on `company`: {a1,a2}, {a3}."""
    return _dedupe_on(source("crn"), "company").collect()


@pytest.fixture
def by_town(source: Callable[..., Source]) -> Resolver:
    """Crn deduplicated on `town`: {a1,a3}, {a2}."""
    return _dedupe_on(source("crn"), "town").collect()


def _leaves(resolver: Resolver) -> dict[str, int]:
    return {
        row["key"]: int(row["leaf"])
        for row in resolver.entities().iter_rows(named=True)
    }


# -- merged sampling ------------------------------------------------------------------


def test_sample_one_resolver(by_company: Resolver) -> None:
    """One resolver samples its own clusters: a pair and a singleton."""
    samples = get_samples(n=999, resolver=by_company)

    assert sorted(len(item.leaves) for item in samples.values()) == [1, 2]


def test_sample_merged(by_company: Resolver, by_town: Resolver) -> None:
    """Two records land together in the union if *either* resolver put them there.

    Individually each resolver splits the source into a pair and a singleton, and
    they disagree about which. Merged, everything is connected. That is the point.
    Every cluster where the two could differ is on screen, so one judgement settles it
    for both. Neither resolver chooses the clusters it is scored on.
    """
    assert sorted(len(s.leaves) for s in get_samples(999, by_company).values()) == [
        1,
        2,
    ]
    assert sorted(len(s.leaves) for s in get_samples(999, by_town).values()) == [1, 2]

    merged = get_samples(n=999, resolver=[by_company, by_town])

    assert len(merged) == 1
    item = next(iter(merged.values()))
    assert sorted(item.leaves) == sorted(_leaves(by_company).values())


def test_sample_merged_content_addressed(
    by_company: Resolver, by_town: Resolver
) -> None:
    """Two people running the same comparison key on the same IDs."""
    once = get_samples(n=999, resolver=[by_company, by_town])
    twice = get_samples(n=999, resolver=[by_town, by_company])

    assert set(once) == set(twice)


def test_sample_rejects_mismatched_sources(
    source: Callable[..., Source],
) -> None:
    """Same source *name*, different data, so the clusters are not comparable.

    A name repeats across generations of a source. Agreeing on the name does not mean
    agreeing on the records.
    """
    wide = _dedupe_on(source("crn"), "company").collect()
    narrow = _dedupe_on(
        source("crn", "select pk, company from crn"), "company"
    ).collect()

    with pytest.raises(SourceTableError, match="disagree about source 'crn'"):
        get_samples(n=999, resolver=[wide, narrow])


def test_sample_empty_sequence_raises(by_company: Resolver) -> None:
    """Sampling no resolvers at all is an error, not an empty draw."""
    with pytest.raises(ValueError, match="At least one resolver"):
        get_samples(n=1, resolver=[])


# -- seeding --------------------------------------------------------------------------


def test_seed_fixes_sample(by_company: Resolver) -> None:
    """What replaced passing a dumped sample file around."""
    first = get_samples(n=1, resolver=by_company, seed=7)
    again = get_samples(n=1, resolver=by_company, seed=7)
    assert set(first) == set(again)


def test_seed_fixes_merged(by_company: Resolver, by_town: Resolver) -> None:
    """A seed fixes the merged sample as it fixes a single one."""
    merged = [by_company, by_town]
    assert set(get_samples(n=1, resolver=merged, seed=7)) == set(
        get_samples(n=1, resolver=merged, seed=7)
    )


# -- scoring --------------------------------------------------------------------------


@pytest.fixture
def judged(by_company: Resolver, store: DuckDBStore) -> DuckDBStore:
    """A judgement saying a1 and a2 are the same company, and a3 is not.

    Deduplicating on `company` reached exactly this conclusion. Deduplicating on
    `town` did not, because it grouped a1 with a3 instead.
    """
    leaves = _leaves(by_company)
    store.store_judgement(
        Judgement(
            tag="bakeoff",
            shown=sorted(leaves.values()),
            endorsed=[sorted([leaves["a1"], leaves["a2"]]), [leaves["a3"]]],
        )
    )
    return store


def test_score_one_resolver(judged: DuckDBStore, by_company: Resolver) -> None:
    """One resolver scores to a single precision/recall pair."""
    precision, recall = EvalData(judged, tag="bakeoff").precision_recall(by_company)

    assert (precision, recall) == (1.0, 1.0)


def test_score_several_resolvers(
    judged: DuckDBStore, by_company: Resolver, by_town: Resolver
) -> None:
    """Ranked against one another over the same records, in the order given."""
    scores = EvalData(judged, tag="bakeoff").precision_recall([by_company, by_town])

    assert scores == [(1.0, 1.0), (0.0, 0.0)]


def test_score_by_label(judged: DuckDBStore, by_company: Resolver) -> None:
    """Scoring a label matches scoring the resolver it points at."""
    by_company.publish("companies")
    evaluation = EvalData(judged, tag="bakeoff")

    assert evaluation.precision_recall("companies") == evaluation.precision_recall(
        by_company
    )


def test_fixture_guard_resolvers_differ(
    by_company: Resolver, by_town: Resolver
) -> None:
    """Guards the fixtures. The scores above mean nothing if these agree."""
    assert by_company.leaf_sets() != by_town.leaf_sets()
    # Also check they cover the same records, or the comparison would not be fair.
    assert set(_leaves(by_company).values()) == set(_leaves(by_town).values())
