"""The worked comparison in `examples/companies` must keep agreeing with itself.

`by_hand.py` exists to be a fair opponent. If matchlab's answer drifts from a
straightforward Polars implementation of the same matching rules, one of them is
wrong, and the comparison in the README stops meaning anything.
"""

import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "companies"


@pytest.fixture(autouse=True)
def examples_on_path() -> None:
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))


@pytest.fixture
def built_warehouse(
    examples_on_path: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Build the warehouse in a temp directory.

    The examples write theirs next to the scripts, which is right for a user reading
    them and wrong for a test suite running in parallel.
    """
    import by_hand  # noqa: PLC0415
    import warehouse  # noqa: PLC0415
    import with_matchlab  # noqa: PLC0415

    db = tmp_path / "warehouse.sqlite"
    for module in (warehouse, by_hand, with_matchlab):
        monkeypatch.setattr(module, "DB", db)
    warehouse.build()


def _as_groups(mapping: dict) -> set[frozenset[str]]:
    return {frozenset(keys) for keys in mapping.values()}


def test_both_pipelines_resolve_identically(built_warehouse: None) -> None:
    import by_hand  # noqa: PLC0415
    import with_matchlab  # noqa: PLC0415

    from matchlab.adapters import DuckDBAdapter  # noqa: PLC0415

    resolution = with_matchlab.build().collect(DuckDBAdapter(":memory:")).resolution()
    from_matchlab: dict[int, set[str]] = {}
    for row in resolution.iter_rows(named=True):
        from_matchlab.setdefault(row["root"], set()).add(row["key"])

    assert _as_groups(from_matchlab) == _as_groups(by_hand.resolve())


def test_the_answer_is_precise_against_known_truth(built_warehouse: None) -> None:
    """Every group is one real company — the pipeline never merges two.

    It does *under*-merge: 'Theta Retail' and 'Theta Retail Group Ltd' clean to
    different names, so exact matching can't join them. That's a matching limitation
    the example is honest about, not a pipeline bug.
    """
    import by_hand  # noqa: PLC0415
    import warehouse  # noqa: PLC0415

    truth = warehouse.truth()
    groups = by_hand.resolve()

    for keys in groups.values():
        assert len({truth[key] for key in keys}) == 1, keys

    assert len(groups) == 9  # 8 true companies, one of them split in two
