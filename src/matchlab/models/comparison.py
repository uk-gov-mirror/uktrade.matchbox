"""Functions to compare fields in different sources."""

import sqlglot.expressions as exp
from sqlglot import parse_one
from sqlglot.errors import ParseError

from matchlab.core.logging import logger
from matchlab.core.sql import SQLCondition


def comparison(sql_condition: SQLCondition, dialect: str = "duckdb") -> SQLCondition:
    """Validate a SQL WHERE-clause condition and recompile it for `dialect`.

    Every column must be qualified with the alias `l` or `r`, the two sides a
    [`Model`][matchlab.models.Model] compares. Raises `ParseError` if a column isn't
    qualified this way, if the condition isn't a valid boolean expression, or if it
    doesn't reference both sides.

    `dialect` only changes how the validated condition renders back to SQL, not how
    it's parsed, so `sql_condition` must already be syntax `sqlglot` can parse
    without help.

    Logs a warning for `OR` conditions, since DuckDB can't use an index to prune them.
    """
    parsed_sql = parse_one(sql_condition)

    for node in parsed_sql.walk():
        # `walk()` yields each expression directly (older sqlglot yielded a
        # `(node, parent, key)` tuple, which is why this once read `node[0]`).
        if not isinstance(
            node,
            exp.Connector | exp.Predicate | exp.Condition | exp.Identifier,
        ):
            raise ParseError(
                f"Must be valid WHERE clause statements. Found {type(node)}"
            )

        if isinstance(node, exp.Or):
            logger.warning(
                "Or conditions are computationally expensive. Expect slow performance."
            )

    left = False
    right = False
    for column in parsed_sql.find_all(exp.Column):
        if column.table == "l":
            left = True
        elif column.table == "r":
            right = True
        elif column.table == "":
            raise ParseError(
                "Columns must be explicitly declared as one of table "
                '"l" or "r".\n\n'
                f"No table declared for column {column.name}."
            )
        else:
            raise ParseError(
                "Columns must be explicitly declared as one of table "
                '"l" or "r".\n\n'
                f"Found column {column.table}."
            )
    if not (left and right):
        raise ParseError('Conditions must reference both "l" and "r".')

    pg_sql = parsed_sql.sql(dialect=dialect)

    return pg_sql
