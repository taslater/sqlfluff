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
        pytest.param("CREATE SCHEMA IF NOT s;", id="if_not_without_exists"),
        pytest.param("CREATE SCHEMA IF EXISTS s;", id="if_exists_without_not"),
        pytest.param("CREATE SCHEMA s COMMENT;", id="comment_without_text"),
        pytest.param("CREATE SCHEMA s DEFAULT COLLATION;", id="default_collation_without_name"),
        pytest.param("CREATE SCHEMA s LOCATION;", id="location_without_path"),
        pytest.param("CREATE SCHEMA s MANAGED LOCATION;", id="managed_location_without_path"),
        pytest.param("CREATE SCHEMA s RETAIN DROPPED FOR 14;", id="retain_dropped_without_unit"),
        pytest.param("CREATE SCHEMA s RETAIN DROPPED FOR DAYS;", id="retain_dropped_without_number"),
        pytest.param("CREATE SCHEMA s WITH DBPROPERTIES ();", id="empty_dbproperties"),
        pytest.param("CREATE SCHEMA s WITH DBPROPERTIES (k =);", id="dbproperties_without_value"),
    ],
)
def test_create_schema_rejections(sql: str) -> None:
    """CREATE SCHEMA clause boundaries."""
    assert _violations(sql), f"Expected violations but got none for:\n{sql}"
