"""Databricks dialect-specific parser rejection tests."""

import pytest

from sqlfluff.core import Linter


def _violations(sql: str) -> list:
    """Return all parse errors, including unparsable nodes in the tree."""
    parsed = Linter(dialect="databricks").parse_string(sql)
    violations: list = list(parsed.violations)
    if parsed.tree:
        violations += list(parsed.tree.recursive_crawl("unparsable"))
    return violations


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param(
            """CREATE MATERIALIZED VIEW bad_mv (
                CONSTRAINT c EXPECT (value > 0),
                value INT
            ) AS SELECT 1 AS value;""",
            id="expectation_before_column",
        ),
        pytest.param(
            """CREATE MATERIALIZED VIEW bad_mv (
                value INT,
                CONSTRAINT pk PRIMARY KEY (value),
                CONSTRAINT c EXPECT (value > 0)
            ) AS SELECT 1 AS value;""",
            id="expectation_after_table_constraint",
        ),
    ],
)
def test_materialized_view_constraints_reject_invalid_order(sql: str) -> None:
    """Materialized view constraints must follow columns and expectations."""
    assert _violations(sql), f"Expected violations but got none for:\n{sql}"


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param(
            "CREATE PRIVATE TABLE t (a INT);",
            id="private_without_streaming",
        ),
        pytest.param(
            "CREATE OR REFRESH PRIVATE TABLE t (a INT);",
            id="private_refresh_without_streaming",
        ),
        pytest.param(
            "CREATE PRIVATE LIVE TABLE t (a INT);",
            id="private_live_without_streaming",
        ),
    ],
)
def test_private_requires_streaming_table(sql: str) -> None:
    """PRIVATE is only valid on a streaming table, not on a table."""
    assert _violations(sql), f"Expected violations but got none for:\n{sql}"


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param(
            "CREATE FUNCTION f() RETURNS INT CONTAINS SQL READS SQL DATA RETURN 1;",
            id="contains_sql_and_reads_sql_data",
        ),
        pytest.param(
            "CREATE FUNCTION f() RETURNS INT RETURN 1 AS $$ return 1 $$;",
            id="body_return_and_as",
        ),
        pytest.param(
            "CREATE FUNCTION f() RETURNS INT LANGUAGE RETURN 1;",
            id="language_without_name",
        ),
        pytest.param(
            "CREATE FUNCTION f() RETURNS INT DEFAULT COLLATION RETURN 1;",
            id="default_collation_without_name",
        ),
        pytest.param(
            "CREATE FUNCTION f() RETURNS INT LANGUAGE PYTHON ENVIRONMENT () AS $$ return 1 $$;",
            id="empty_environment",
        ),
        pytest.param(
            "CREATE FUNCTION f() RETURNS INT LANGUAGE PYTHON ENVIRONMENT (dependencies =) AS $$ return 1 $$;",
            id="environment_without_value",
        ),
        pytest.param(
            "CREATE FUNCTION f() RETURNS INT;",
            id="without_body",
        ),
    ],
)
def test_create_function_characteristic_rejections(sql: str) -> None:
    """CREATE FUNCTION characteristic boundaries."""
    assert _violations(sql), f"Expected violations but got none for:\n{sql}"
