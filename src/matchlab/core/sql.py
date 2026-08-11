"""SQL fragment string types shared across matchlab."""

from typing import TypeAlias

SQLExpression: TypeAlias = str
"""Any SQL text `sqlglot` can parse, with no shape guaranteed beyond that."""

SQLCondition: TypeAlias = SQLExpression
"""A boolean WHERE-clause predicate, qualified with `l`/`r` per `models.comparison`."""

SQLQuery: TypeAlias = SQLExpression
"""A full SQL statement, such as a `SELECT`, not just a predicate."""
