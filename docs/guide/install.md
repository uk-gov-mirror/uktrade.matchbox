# Install

```bash
pip install matchlab
```

matchlab is a library, not a service. There is nothing to deploy, no credentials to configure, and no environment variables to set before you can use it.

## What you need

**A warehouse to read from.** matchlab reads source data through a [`Location`](../api/locations.md). Anything SQLAlchemy or ADBC can connect to works, including Postgres, SQLite, and DuckDB.

```python
from sqlalchemy import create_engine
from matchlab import RelationalDBLocation

warehouse = RelationalDBLocation(
    name="warehouse",
    client=create_engine("postgresql://user:pass@localhost:5432/db"),
)
```

**Somewhere to keep results.** matchlab stores what it computes in an [adapter](../api/adapters.md). By default, that's a DuckDB file in your user cache directory, created on first use. You don't have to do anything.

To put it somewhere specific:

```python
from matchlab import set_default_adapter
from matchlab.adapters import DuckDBAdapter

set_default_adapter(DuckDBAdapter("./pipeline.duckdb"))
```

Or per collection, without changing the default:

```python
plan.collect(adapter=DuckDBAdapter("./scratch.duckdb"))
```

An in-memory store (`DuckDBAdapter(":memory:")`) is useful in tests.

## Verify

```python
import matchlab

print(matchlab.__version__)
```

## Next

[Build a plan :octicons-arrow-right-16:](./build-a-plan.md){ .md-button .md-button--primary }
