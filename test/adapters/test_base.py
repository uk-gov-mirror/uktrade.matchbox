"""The stats helpers on the `Adapter` base: `format_bytes` and `StoreStats.describe`.

These are pure formatting over plain values — no store, no backend — so they belong to
`base.py`'s unit, not to a round-trip through a DuckDB file.
"""

import pytest

from matchlab.adapters import StoreStats, format_bytes
from matchlab.core.kinds import StepKind


@pytest.mark.parametrize(
    ("count", "signed", "expected"),
    [
        pytest.param(0, False, "0 B", id="zero"),
        pytest.param(0, True, "+0 B", id="signed-zero"),
        pytest.param(512, False, "512 B", id="bytes"),
        pytest.param(1024, False, "1.0 KB", id="kilobytes"),
        pytest.param(1536, True, "+1.5 KB", id="signed-growth"),
        pytest.param(5 * 1024**3, False, "5.0 GB", id="gigabytes"),
        # A store only shrinks if something rewrites it.
        pytest.param(-2048, True, "-2.0 KB", id="signed-shrink"),
    ],
)
def test_format_bytes(count: int, signed: bool, expected: str) -> None:
    assert format_bytes(count, signed=signed) == expected


def test_stats_describe_themselves() -> None:
    stats = StoreStats(location="somewhere", bytes=2048, artifacts={StepKind.VIEW: 3})

    assert stats.describe() == "Store 2.0 KB, 3 artifacts"
    assert (
        stats.describe(since=StoreStats(location="somewhere", bytes=1024))
        == "Store 2.0 KB (+1.0 KB), 3 artifacts"
    )


def test_an_empty_store_describes_itself_without_a_plural() -> None:
    one = StoreStats(location="x", bytes=0, artifacts={StepKind.SOURCE: 1})

    assert StoreStats(location="x", bytes=0).describe() == "Store 0 B, 0 artifacts"
    assert one.describe() == "Store 0 B, 1 artifact"
