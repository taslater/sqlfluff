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
        pytest.param("SHOW GRANTS;\n", id="no_securable"),
        pytest.param("SHOW GRANTS TABLE my_table;\n", id="missing_on"),
        pytest.param("SHOW GRANTS `alf` my_table;\n", id="principal_without_on"),
    ],
)
def test_show_grants_requires_a_securable(sql: str) -> None:
    """`SHOW GRANTS [ principal ] ON securable_object`.

    The securable and its `ON` are both required; only the principal is
    optional.
    """
    assert _violations(sql), f"Expected a parse failure for:\n{sql}"


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param("CREATE OR REFRESH VIEW v AS SELECT 1;\n", id="plain_view"),
        pytest.param("CREATE OR REFRESH LIVE VIEW v AS SELECT 1;\n", id="live_view"),
        pytest.param(
            "CREATE OR REFRESH TEMPORARY STREAMING LIVE VIEW v AS SELECT 1;\n",
            id="streaming_live_view",
        ),
    ],
)
def test_or_refresh_is_not_a_view_clause(sql: str) -> None:
    """OR REFRESH belongs to streaming tables and materialized views.

    The corpus of published Databricks SQL uses it with MATERIALIZED VIEW,
    STREAMING TABLE and LIVE TABLE, and with no VIEW form at all. Both of
    those statements have their own segments here, so CREATE VIEW does not
    need it.
    """
    assert _violations(sql), f"Expected a parse failure for:\n{sql}"


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param(
            "SHOW GRANTS ON STORAGE sc;\n",
            id="storage_without_credential",
        ),
        pytest.param(
            "SHOW GRANTS ON STORAGE CREDENTIAL;\n",
            id="credential_without_name",
        ),
        pytest.param(
            "SHOW GRANTS ON SERVICE sc;\n",
            id="service_without_credential",
        ),
        pytest.param(
            "GRANT ALL PRIVILEGES, SELECT ON TABLE t TO p;\n",
            id="all_privileges_in_list",
        ),
        pytest.param(
            "GRANT SELECT, ALL PRIVILEGES ON TABLE t TO p;\n",
            id="all_privileges_later_in_list",
        ),
        pytest.param(
            "GRANT ALL, SELECT ON TABLE t TO p;\n",
            id="bare_all_in_list",
        ),
    ],
)
def test_privileges_bind_their_clauses(sql: str) -> None:
    """Privilege forms bind, in both directions.

    A credential is `[ STORAGE | SERVICE ] CREDENTIAL name`, never the scope
    keyword alone or without a name, and `privilege_types` is
    `{ ALL PRIVILEGES | privilege_type [, ...] }`, so ALL PRIVILEGES cannot
    open or join a list.
    """
    assert _violations(sql), f"Expected violations but got none for:\n{sql}"


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param(
            "CREATE TEMPORARY VIEW v USING;\n",
            id="using_without_data_source",
        ),
        pytest.param(
            "CREATE VIEW v USING csv OPTIONS (path '/data');\n",
            id="using_without_temporary",
        ),
        pytest.param(
            "CREATE TEMPORARY VIEW v USING csv OPTIONS ();\n",
            id="empty_options",
        ),
        pytest.param(
            "CREATE TEMPORARY VIEW v USING csv OPTIONS (path);\n",
            id="options_without_value",
        ),
        pytest.param(
            "CREATE VIEW v WITH AS SELECT a FROM t;\n",
            id="with_without_clause",
        ),
    ],
)
def test_view_requires_bound_clauses(sql: str) -> None:
    """The data-source production and the with_clause bind their tokens.

    `USING` needs a data source, the data-source production is only for a
    TEMPORARY view, `OPTIONS` needs at least one name-value pair, and `WITH`
    needs a clause.
    """
    assert _violations(sql), f"Expected violations but got none for:\n{sql}"


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param(
            "CREATE OR REFRESH STREAMING TABLE t FLOW INSERT SELECT * FROM STREAM s;",
            id="flow_insert_without_by_name",
        ),
        pytest.param(
            "CREATE OR REFRESH STREAMING TABLE t "
            "FLOW REPLACE USING (c) BY NAME SELECT * FROM STREAM s;",
            id="flow_replace_using_without_sequence_by",
        ),
        pytest.param(
            "CREATE OR REFRESH STREAMING TABLE t "
            "FLOW REPLACE USING (c) SEQUENCE BY d SELECT * FROM STREAM s;",
            id="flow_replace_using_without_by_name",
        ),
        pytest.param(
            "CREATE OR REFRESH STREAMING TABLE t "
            "FLOW SEQUENCE BY d BY NAME SELECT * FROM STREAM s;",
            id="flow_sequence_by_without_replace_using",
        ),
        pytest.param(
            "CREATE TABLE t FLOW INSERT BY NAME SELECT * FROM STREAM s;",
            id="flow_without_streaming",
        ),
        pytest.param(
            "CREATE PRIVATE TABLE t FLOW INSERT BY NAME SELECT * FROM STREAM s;",
            id="flow_with_private_without_streaming",
        ),
    ],
)
def test_inline_flow_requires_streaming_and_a_bound_spec(sql: str) -> None:
    """An inline FLOW is only valid on a streaming table, and its spec binds.

    `REPLACE USING (...)` and `SEQUENCE BY` are required together, and the
    append form takes `BY NAME`. These are the boundaries #8509 missed for the
    standalone statement, so they are asserted here rather than left to the
    fixture, which cannot express a rejection.
    """
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
