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
            "CREATE CATALOG c USING SHARE provider;\n",
            id="share_without_share_name",
        ),
        pytest.param(
            "CREATE CATALOG c RETAIN DROPPED FOR 30;\n",
            id="retain_dropped_without_unit",
        ),
        pytest.param(
            "CREATE CATALOG c DEFAULT COLLATION;\n",
            id="default_collation_without_name",
        ),
        pytest.param(
            "CREATE CATALOG c OPTIONS ();\n",
            id="empty_options",
        ),
        pytest.param(
            "CREATE CATALOG c OPTIONS (k =);\n",
            id="options_without_value",
        ),
        pytest.param(
            "CREATE FOREIGN CATALOG fc OPTIONS (k = 'v');\n",
            id="foreign_without_connection",
        ),
        pytest.param(
            "CREATE FOREIGN CATALOG fc USING CONNECTION conn;\n",
            id="foreign_without_options",
        ),
    ],
)
def test_create_catalog_requires_bound_clauses(sql: str) -> None:
    """A clause's required tokens are all required.

    The reference gives the clause list as a bracketed alternation: a share
    needs both name parts, RETAIN DROPPED takes a number and a unit, a
    collation needs its name, OPTIONS needs at least one name-value pair, and
    the foreign form needs both USING CONNECTION and OPTIONS.
    """
    assert _violations(sql), f"Expected violations but got none for:\n{sql}"
