"""Direct unit tests for `SplinkLinker` behavior.

These tests cover Splink-specific schema handling and diagnostics, rather than the
shared match/no-match behavior exercised in the other linker suites.
"""

import polars as pl
import pytest
from splink import SettingsCreator
from splink import blocking_rule_library as brl
from splink import comparison_library as cl

from matchlab.models.linkers.splinklinker import SplinkLinker


def test_splink_conformancy_error() -> None:
    """Splink surfaces the concrete schema mismatch from the real prepare path."""
    linker = SplinkLinker(
        left_id="id",
        right_id="id",
        linker_training_functions=[],
        linker_settings=SettingsCreator(
            link_type="link_only",
            blocking_rules_to_generate_predictions=[brl.block_on("id")],
            comparisons=[cl.ExactMatch("id")],
        ),
        threshold=None,
    )
    left = pl.DataFrame({"id": [1], "company_name": ["left"]})
    right = pl.DataFrame({"id": [2], "postcode": ["SW1A 1AA"]})

    with pytest.raises(
        ValueError,
        match="SplinkLinker requires input data to be conformant",
    ) as exc_info:
        linker.prepare(left, right)

    message = str(exc_info.value)
    assert "Left schema: id=Int64, company_name=String" in message
    assert "Right schema: id=Int64, postcode=String" in message
    assert "Columns only on left: ['company_name']" in message
    assert "Columns only on right: ['postcode']" in message
