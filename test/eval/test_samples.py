"""Sampling and scoring across more than one resolution.

Scenario — one source, two ways of deduplicating it:

    crn (pk, company, town):  a1=(acme,london) a2=(acme,leeds) a3=(beta,london)

Dedupe on `company` and you get {a1,a2}, {a3}. Dedupe on `town` and you get {a1,a3},
{a2}. The two disagree about every record, which is exactly when you want to judge them
against the same clusters.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine, create_engine, text

from matchlab import Resolver, Source, set_default_adapter
from matchlab.adapters import DuckDBAdapter
from matchlab.core.exceptions import SourceTableError
from matchlab.eval import EvalData, get_samples
from matchlab.eval.judgements import Judgement
from matchlab.locations import RelationalDBLocation
from matchlab.models.dedupers import NaiveDeduper


@pytest.fixture
def warehouse(tmp_path: Path) -> Engine:
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


@pytest.fixture(autouse=True)
def adapter() -> Iterator[DuckDBAdapter]:
    store = DuckDBAdapter(":memory:")
    set_default_adapter(store)
    yield store
    set_default_adapter(None)
    store.close()


def _source(
    warehouse: Engine, extract: str = "select pk, company, town from crn"
) -> Source:
    return Source(
        location=RelationalDBLocation(name="warehouse", client=warehouse),
        name="crn",
        extract_transform=extract,
        key_field="pk",
    )


def _dedupe_on(source: Source, field: str) -> Resolver:
    return source.dedupe(
        model_class=NaiveDeduper,
        model_settings={"unique_fields": [source.f(field)]},
    ).resolve()


@pytest.fixture
def by_company(warehouse: Engine) -> Resolver:
    return _dedupe_on(_source(warehouse), "company").collect()


@pytest.fixture
def by_town(warehouse: Engine) -> Resolver:
    return _dedupe_on(_source(warehouse), "town").collect()


def _leaves(resolver: Resolver) -> dict[str, int]:
    return {
        row["key"]: int(row["leaf"])
        for row in resolver.entities().iter_rows(named=True)
    }


# -- merged sampling ------------------------------------------------------------------


def test_one_resolution_samples_its_own_clusters(by_company: Resolver) -> None:
    samples = get_samples(n=999, resolution=by_company)

    assert sorted(len(item.leaves) for item in samples.values()) == [1, 2]


def test_several_resolutions_sample_their_merged_clusters(
    by_company: Resolver, by_town: Resolver
) -> None:
    """The union: two records land together if *either* resolution put them there.

    Individually each resolution splits the source into a pair and a singleton, and
    they disagree about which. Merged, everything is connected — which is the point:
    every cluster where the two could differ is on screen, so one judgement settles it
    for both, and neither gets to choose the clusters it is scored on.
    """
    assert sorted(len(s.leaves) for s in get_samples(999, by_company).values()) == [
        1,
        2,
    ]
    assert sorted(len(s.leaves) for s in get_samples(999, by_town).values()) == [1, 2]

    merged = get_samples(n=999, resolution=[by_company, by_town])

    assert len(merged) == 1
    item = next(iter(merged.values()))
    assert sorted(item.leaves) == sorted(_leaves(by_company).values())


def test_merged_roots_are_content_addressed(
    by_company: Resolver, by_town: Resolver
) -> None:
    """Two people running the same comparison key on the same IDs."""
    once = get_samples(n=999, resolution=[by_company, by_town])
    twice = get_samples(n=999, resolution=[by_town, by_company])

    assert set(once) == set(twice)


def test_resolutions_over_different_source_artifacts_are_rejected(
    warehouse: Engine,
) -> None:
    """Same source *name*, different data — so the clusters are not comparable.

    A name repeats across generations of a source; agreeing on it is not agreeing on
    the records.
    """
    wide = _dedupe_on(_source(warehouse), "company").collect()
    narrow = _dedupe_on(
        _source(warehouse, "select pk, company from crn"), "company"
    ).collect()

    with pytest.raises(SourceTableError, match="disagree about source 'crn'"):
        get_samples(n=999, resolution=[wide, narrow])


def test_an_empty_sequence_is_an_error(by_company: Resolver) -> None:
    with pytest.raises(ValueError, match="At least one resolution"):
        get_samples(n=1, resolution=[])


# -- seeding --------------------------------------------------------------------------


def test_a_seed_fixes_which_clusters_come_back(by_company: Resolver) -> None:
    """What replaced passing a dumped sample file around."""
    first = get_samples(n=1, resolution=by_company, seed=7)
    again = get_samples(n=1, resolution=by_company, seed=7)
    assert set(first) == set(again)


def test_a_seed_fixes_the_merged_sample_too(
    by_company: Resolver, by_town: Resolver
) -> None:
    merged = [by_company, by_town]
    assert set(get_samples(n=1, resolution=merged, seed=7)) == set(
        get_samples(n=1, resolution=merged, seed=7)
    )


# -- scoring --------------------------------------------------------------------------


@pytest.fixture
def judged(by_company: Resolver, adapter: DuckDBAdapter) -> DuckDBAdapter:
    """A judgement saying a1 and a2 are the same company, and a3 is not.

    Which is exactly what deduplicating on `company` concluded, and exactly what
    deduplicating on `town` — which grouped a1 with a3 — did not.
    """
    leaves = _leaves(by_company)
    adapter.store_judgement(
        Judgement(
            tag="bakeoff",
            shown=sorted(leaves.values()),
            endorsed=[sorted([leaves["a1"], leaves["a2"]]), [leaves["a3"]]],
        )
    )
    return adapter


def test_one_resolution_scores_to_a_pair(
    judged: DuckDBAdapter, by_company: Resolver
) -> None:
    precision, recall = EvalData(judged, tag="bakeoff").precision_recall(by_company)

    assert (precision, recall) == (1.0, 1.0)


def test_several_resolutions_score_to_a_list(
    judged: DuckDBAdapter, by_company: Resolver, by_town: Resolver
) -> None:
    """Ranked against one another over the same records, in the order given."""
    scores = EvalData(judged, tag="bakeoff").precision_recall([by_company, by_town])

    assert scores == [(1.0, 1.0), (0.0, 0.0)]


def test_a_label_scores_the_same_as_the_resolver_it_names(
    judged: DuckDBAdapter, by_company: Resolver
) -> None:
    by_company.publish("companies")
    evaluation = EvalData(judged, tag="bakeoff")

    assert evaluation.precision_recall("companies") == evaluation.precision_recall(
        by_company
    )


def test_entities_of_the_two_resolutions_really_do_differ(
    by_company: Resolver, by_town: Resolver
) -> None:
    """Guards the fixtures: the scores above mean nothing if these agree."""
    assert by_company.leaf_sets() != by_town.leaf_sets()
    # And over the same records, or the comparison would not be fair.
    assert set(_leaves(by_company).values()) == set(_leaves(by_town).values())
