"""Methodology tests build models directly, so they need somewhere to collect into."""

from collections.abc import Iterator

import pytest

from matchlab.adapters import DuckDBAdapter
from matchlab.steps import set_default_adapter


@pytest.fixture(autouse=True)
def adapter() -> Iterator[DuckDBAdapter]:
    store = DuckDBAdapter(":memory:")
    set_default_adapter(store)
    yield store
    set_default_adapter(None)
    store.close()
