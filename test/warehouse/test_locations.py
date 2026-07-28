"""Location behaviour across warehouse client types.

Parametrised over SQLAlchemy and ADBC clients — several tests exist specifically to
compare dialect-specific extract/transform validation. Fixtures live in `conftest.py`;
none of them needs Docker.
"""

from typing import TypeAlias

import polars as pl
import pytest
from adbc_driver_manager import ProgrammingError
from adbc_driver_manager.dbapi import Connection as AdbcConnection
from polars.testing import assert_frame_equal
from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError

from matchlab.core.exceptions import ExtractTransformError
from matchlab.core.factories.sources import (
    FeatureConfig,
    source_factory,
)
from matchlab.locations import (
    ClientType,
    RelationalDBLocation,
    add_location_class,
    build_location,
)

#: What the `warehouse` fixture hands back — the client types a location accepts.
WarehouseClient: TypeAlias = Engine | AdbcConnection


@pytest.mark.parametrize(
    ("warehouse", "expected_client_type"),
    [
        pytest.param("sqla_sqlite", ClientType.SQLALCHEMY, id="sqlalchemy"),
        pytest.param("adbc_sqlite", ClientType.ADBC, id="adbc"),
    ],
    indirect=["warehouse"],
)
def test_relational_db_location_instantiation(
    warehouse: WarehouseClient,
    expected_client_type: ClientType,
) -> None:
    """A location is named and clientful from the moment it exists."""
    location = RelationalDBLocation(name="dbname", client=warehouse)

    assert location.name == "dbname"
    assert location.client is warehouse
    assert location.client_type == expected_client_type


def test_a_location_rejects_a_client_it_cannot_drive(
    sqla_sqlite_warehouse: Engine,
) -> None:
    """Validation is the subclass's, so it knows what it can actually use."""
    with pytest.raises(ValueError, match="not a valid client"):
        RelationalDBLocation(name="dbname", client=12)

    # And a client cannot be swapped afterwards, which would silently invalidate
    # anything already validated against the old one's dialect.
    location = RelationalDBLocation(name="dbname", client=sqla_sqlite_warehouse)
    with pytest.raises(AttributeError):
        location.client = sqla_sqlite_warehouse


def test_a_location_class_is_found_by_its_registered_name(
    sqla_sqlite_warehouse: Engine,
) -> None:
    """Documents name location classes the way they name dedupers and resolvers."""
    rebuilt = build_location(
        "RelationalDBLocation", name="elsewhere", client=sqla_sqlite_warehouse
    )
    assert isinstance(rebuilt, RelationalDBLocation)
    assert rebuilt.name == "elsewhere"
    assert rebuilt.client is sqla_sqlite_warehouse

    with pytest.raises(ValueError, match="No location class named 'Nowhere'"):
        build_location("Nowhere", name="x", client=sqla_sqlite_warehouse)


def test_a_custom_location_class_can_be_registered(
    sqla_sqlite_warehouse: Engine,
) -> None:
    """The registry is open, so a document can travel to a codebase we don't ship."""

    class CustomLocation(RelationalDBLocation):
        pass

    with pytest.raises(ValueError, match="not a subclass of Location"):
        add_location_class(int)

    add_location_class(CustomLocation)
    assert isinstance(
        build_location("CustomLocation", name="dbname", client=sqla_sqlite_warehouse),
        CustomLocation,
    )


@pytest.mark.parametrize(
    "warehouse",
    ["sqla_sqlite", "adbc_sqlite"],
    indirect=True,
)
def test_relational_db_connect(warehouse: WarehouseClient) -> None:
    """Test connecting to database."""
    location = RelationalDBLocation(name="dbname", client=warehouse)
    assert location.connect() is True


@pytest.mark.parametrize(
    ("sql", "dialects"),
    [
        pytest.param("SELECT * FROM test_table", "all", id="valid-select"),
        pytest.param(
            "SELECT id, name FROM test_table WHERE id > 1", "all", id="valid-where"
        ),
        pytest.param("SLECT * FROM test_table", "none", id="invalid-syntax"),
        pytest.param("", "none", id="empty-string"),
        pytest.param("ALTER TABLE test_table", "none", id="alter-sql"),
        pytest.param(
            "INSERT INTO users (name, age) VALUES ('John', '25')",
            "none",
            id="insert-sql",
        ),
        pytest.param("DROP TABLE test_table", "none", id="drop-sql"),
        pytest.param("SELECT * FROM users /* with a comment */", "all", id="comment"),
        pytest.param(
            "WITH user_cte AS (SELECT * FROM users) SELECT * FROM user_cte",
            "all",
            id="valid-with",
        ),
        pytest.param(
            (
                "WITH user_cte AS (SELECT * FROM users) "
                "INSERT INTO temp_users SELECT * FROM user_cte"
            ),
            "none",
            id="invalid-with",
        ),
        pytest.param(
            "SELECT * FROM users; DROP TABLE users;", "none", id="multiple-statements"
        ),
        pytest.param("SELECT * INTO new_table FROM users", "none", id="select-into"),
        pytest.param(
            """
            WITH updated_rows AS (
                UPDATE employees
                SET salary = salary * 1.1
                WHERE department = 'Sales'
                RETURNING *
            )
            SELECT * FROM updated_rows;
            """,
            "none",
            id="non-query-cte",
        ),
        pytest.param(
            """
            SELECT foo, bar FROM baz
            UNION
            SELECT foo, bar FROM qux;
            """,
            "all",
            id="valid-union",
        ),
        pytest.param(
            """
            SELECT 'ciao' ~ 'hello'
            """,
            "postgres",
            id="postgres-only",
        ),
        pytest.param("""select `name` from user""", "sqlite", id="sqlite-only"),
    ],
)
def test_relational_db_extract_transform(
    sql: str,
    dialects: str,
    sqla_postgres_warehouse: Engine,
    sqla_sqlite_warehouse: Engine,
) -> None:
    """Test SQL validation in validate_extract_transform."""

    if dialects == "none":
        invalid_clients = [sqla_postgres_warehouse, sqla_sqlite_warehouse]
        valid_clients = []
    if dialects == "all":
        invalid_clients = []
        valid_clients = [sqla_postgres_warehouse, sqla_sqlite_warehouse]
    if dialects == "postgres":
        invalid_clients = [sqla_sqlite_warehouse]
        valid_clients = [sqla_postgres_warehouse]
    if dialects == "sqlite":
        invalid_clients = [sqla_postgres_warehouse]
        valid_clients = [sqla_sqlite_warehouse]

    for client in valid_clients:
        RelationalDBLocation(name="dbname", client=client).validate_extract_transform(
            sql
        )

    for client in invalid_clients:
        with pytest.raises(ExtractTransformError):
            RelationalDBLocation(
                name="dbname", client=client
            ).validate_extract_transform(sql)


@pytest.mark.parametrize(
    "warehouse",
    ["sqla_sqlite", "adbc_sqlite"],
    indirect=True,
)
def test_relational_db_execute(
    warehouse: WarehouseClient,
    sqla_sqlite_warehouse: Engine,
) -> None:
    """Test executing a query and returning results using a real SQLite database."""
    features = [
        FeatureConfig(name="company", base_generator="company"),
        FeatureConfig(name="employees", base_generator="random_int"),
    ]

    source_testkit = source_factory(
        features=features, n_true_entities=10, engine=sqla_sqlite_warehouse
    ).write_to_location()

    location = RelationalDBLocation(name="dbname", client=warehouse)

    sql = f"select key, upper(company) up_company, employees from {source_testkit.name}"

    batch_size = 2

    # Execute the query
    results = list(location.execute(sql, batch_size))

    # Right number of batches and total rows
    if isinstance(warehouse, Engine):
        # SQLite ADBC driver doesn't batch, can't cover this
        assert len(results[0]) == batch_size
    combined_df: pl.DataFrame = pl.concat(results)
    assert len(combined_df) == 10

    # Right fields, types and transformations
    assert set(combined_df.columns) == {"key", "up_company", "employees"}
    assert combined_df["employees"].dtype == pl.Int64
    sample_str = combined_df.select("up_company").row(0)[0]
    assert sample_str == sample_str.upper()

    # Try overriding schema
    overridden_results = list(
        location.execute(sql, batch_size, schema_overrides={"employees": pl.String})
    )
    assert overridden_results[0]["employees"].dtype == pl.String

    # Try query with filter
    keys_to_filter = source_testkit.data["key"][:2].to_pylist()
    filtered_results = pl.concat(
        location.execute(sql, batch_size, keys=("key", keys_to_filter))
    )
    assert len(filtered_results) == 2

    # Filtering by no keys has no effect
    unfiltered_results = pl.concat(location.execute(sql, batch_size, keys=("key", [])))
    assert_frame_equal(unfiltered_results, combined_df)


@pytest.mark.parametrize(
    "warehouse",
    ["sqla_sqlite", "adbc_sqlite"],
    indirect=True,
)
def test_relational_db_execute_invalid(warehouse: WarehouseClient) -> None:
    """Test that invalid queries are handled correctly when executing."""
    location = RelationalDBLocation(name="dbname", client=warehouse)

    # Invalid SQL query
    sql = "SELECT * FROM nonexistent_table"

    # Should raise an exception when executed
    with pytest.raises((OperationalError, ProgrammingError)):
        list(location.execute(sql, batch_size=10))
