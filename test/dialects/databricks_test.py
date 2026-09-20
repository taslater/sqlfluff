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
            "COPY INTO t () FROM 'p' FILEFORMAT = CSV;",
            id="empty_column_list",
        ),
        pytest.param(
            "COPY INTO t (a, ) FROM 'p' FILEFORMAT = CSV;",
            id="column_list_trailing_comma",
        ),
        pytest.param(
            "COPY INTO t FROM 'p' FILEFORMAT = CSV VALIDATE 10;",
            id="validate_without_unit",
        ),
        pytest.param(
            "COPY INTO t FROM 'p' FILEFORMAT = CSV VALIDATE ROWS;",
            id="validate_without_number",
        ),
        pytest.param(
            "COPY INTO t FROM 'p' FILEFORMAT = CSV FILES = ();",
            id="empty_files",
        ),
        pytest.param(
            "COPY INTO t FROM 'p' FILEFORMAT = CSV FILES = ('a.csv', );",
            id="file_list_trailing_comma",
        ),
        pytest.param(
            "COPY INTO t FROM 'p' FILEFORMAT = CSV FORMAT_OPTIONS ();",
            id="empty_format_options",
        ),
        pytest.param(
            "COPY INTO t FROM 'p' FILEFORMAT = CSV FORMAT_OPTIONS (header =);",
            id="format_option_without_value",
        ),
        pytest.param(
            "COPY INTO t FROM 'p' FILEFORMAT = CSV COPY_OPTIONS ();",
            id="empty_copy_options",
        ),
        pytest.param(
            "COPY INTO t FROM 'p' FILEFORMAT = CSV COPY_OPTIONS (force =);",
            id="copy_option_without_value",
        ),
        pytest.param(
            "COPY INTO t FROM 'p' FILEFORMAT =;",
            id="fileformat_without_source",
        ),
        pytest.param(
            "COPY INTO t FROM FILEFORMAT = CSV;",
            id="from_without_source",
        ),
        pytest.param(
            "COPY INTO t FROM 'p' WITH (CREDENTIAL) FILEFORMAT = CSV;",
            id="credential_without_name",
        ),
        pytest.param(
            "COPY INTO t FROM 'p' WITH (ENCRYPTION ()) FILEFORMAT = CSV;",
            id="empty_encryption",
        ),
        pytest.param(
            "COPY INTO t FROM 'p' FILEFORMAT = CSV FILES = ('a.csv') PATTERN = '*.csv';",
            id="files_and_pattern",
        ),
    ],
)
def test_copy_into_rejections(sql: str) -> None:
    """COPY INTO clause boundaries that a valid-parse fixture cannot express."""
    assert _violations(sql), f"Expected violations but got none for:\n{sql}"
