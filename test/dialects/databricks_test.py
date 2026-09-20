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
        pytest.param("CREATE CONNECTION TYPE POSTGRESQL OPTIONS (host 'h');", id="connection_without_name"),
        pytest.param("CREATE CONNECTION c OPTIONS (host 'h');", id="connection_without_type"),
        pytest.param("CREATE CONNECTION c TYPE OPTIONS (host 'h');", id="connection_without_type_value"),
        pytest.param("CREATE CONNECTION c TYPE POSTGRESQL;", id="connection_without_options"),
        pytest.param("CREATE CONNECTION c TYPE POSTGRESQL OPTIONS ();", id="connection_empty_options"),
        pytest.param("CREATE CONNECTION c TYPE POSTGRESQL OPTIONS (host);", id="connection_without_option_value"),
        pytest.param("CREATE CONNECTION c TYPE POSTGRESQL OPTIONS (host 'h', );", id="connection_options_trailing_comma"),
        pytest.param("CREATE EXTERNAL LOCATION URL 'u' WITH (STORAGE CREDENTIAL c);", id="location_without_name"),
        pytest.param("CREATE EXTERNAL LOCATION l 'u' WITH (STORAGE CREDENTIAL c);", id="location_without_url_keyword"),
        pytest.param("CREATE EXTERNAL LOCATION l URL WITH (STORAGE CREDENTIAL c);", id="location_without_url_value"),
        pytest.param("CREATE EXTERNAL LOCATION l URL 'u';", id="location_without_with"),
        pytest.param("CREATE EXTERNAL LOCATION l URL 'u' WITH ();", id="location_empty_with"),
        pytest.param("CREATE EXTERNAL LOCATION l URL 'u' WITH (STORAGE CREDENTIAL);", id="location_without_credential_name"),
        pytest.param("CREATE EXTERNAL LOCATION l URL 'u' WITH (STORAGE CREDENTIAL c) COMMENT;", id="location_without_comment_value"),
    ],
)
def test_create_connection_location_rejections(sql: str) -> None:
    """CREATE CONNECTION / EXTERNAL LOCATION boundaries."""
    assert _violations(sql), f"Expected violations but got none for:\n{sql}"
