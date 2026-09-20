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
        pytest.param("SELECT * FROM t AS left;", id="left_table_alias"),
        pytest.param("SELECT * FROM t AS right;", id="right_table_alias"),
    ],
)
def test_left_right_reserved_as_table_aliases(sql: str) -> None:
    """LEFT/RIGHT are unreserved except as unquoted table aliases."""
    assert _violations(sql), f"Expected violations but got none for:\n{sql}"


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param(
            "-- Databricks notebook source\n"
            "-- MAGIC %md\n"
            "-- MAGIC Some prose quoting a command:\n"
            "-- MAGIC %fs ls /tmp/data \n"
            "\n"
            "-- COMMAND ----------\n"
            "\n"
            "SELECT a FROM b;",
            id="magic_line_with_trailing_space",
        ),
        pytest.param(
            "-- Databricks notebook source\n"
            "-- MAGIC %md\n"
            "-- MAGIC Some prose.\n"
            "-- MAGIC %fs\n"
            "\n"
            "-- COMMAND ----------\n"
            "\n"
            "SELECT a FROM b;",
            id="standalone_directive_ends_the_cell",
        ),
    ],
)
def test_magic_cell_boundaries(sql: str) -> None:
    """A magic cell ends at its separator, not at a directive-shaped line.

    A `-- MAGIC` body line may carry a trailing space, and a cell may end with
    a standalone directive; neither may swallow the blank line the command
    separator needs. These are whitespace-sensitive boundaries a `.sql` fixture
    cannot express, because trailing whitespace is stripped from fixtures.
    """
    assert not _violations(sql), f"Expected a clean parse for:\n{sql}"


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
            "CREATE PRIVATE TABLE t FLOW INSERT BY NAME SELECT * FROM STREAM s;",
            id="flow_with_private_without_streaming",
        ),
    ],
)
def test_inline_flow_requires_streaming_and_a_bound_spec(sql: str) -> None:
    """An inline FLOW's spec binds, and PRIVATE still requires STREAMING.

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

@pytest.mark.parametrize(
    "sql",
    [
        pytest.param(
            "ALTER TABLE RENAME TO t2;",
            id="no_table_name",
        ),
        pytest.param(
            "ALTER TABLE t REPLACE PARTITIONED BY WITH;",
            id="replace_partitioned_without_cluster_by",
        ),
    ],
)
def test_alter_table_rejections(sql: str) -> None:
    """ALTER TABLE boundaries that a valid-parse fixture cannot express."""
    assert _violations(sql), f"Expected violations but got none for:\n{sql}"

@pytest.mark.parametrize(
    "sql",
    [
        pytest.param("CREATE SHARE;", id="share_without_name"),
        pytest.param("CREATE SHARE COMMENT 'x';", id="share_without_name_with_comment"),
        pytest.param("CREATE SHARE s COMMENT;", id="share_without_comment_value"),
        pytest.param("CREATE RECIPIENT USING ID 'x';", id="recipient_without_name"),
        pytest.param("CREATE RECIPIENT r USING ID;", id="recipient_without_sharing_id"),
        pytest.param("CREATE RECIPIENT r PROPERTIES ();", id="recipient_empty_properties"),
        pytest.param("CREATE RECIPIENT r PROPERTIES (k =);", id="recipient_without_property_value"),
        pytest.param("CREATE RECIPIENT r COMMENT;", id="recipient_without_comment_value"),
    ],
)
def test_create_share_recipient_rejections(sql: str) -> None:
    """CREATE SHARE / RECIPIENT boundaries a valid-parse fixture cannot express."""
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

@pytest.mark.parametrize(
    "sql",
    [
        pytest.param(
            "CREATE TABLE t (a INT) DEFAULT COLLATION;",
            id="default_collation_without_name",
        ),
        pytest.param(
            "CREATE TABLE t (a INT) LOCATION 'x' WITH (CREDENTIAL);",
            id="location_without_credential_name",
        ),
    ],
)
def test_create_table_clause_rejections(sql: str) -> None:
    """CREATE TABLE clause boundaries."""
    assert _violations(sql), f"Expected violations but got none for:\n{sql}"

@pytest.mark.parametrize(
    "sql",
    [
        pytest.param("OPTIMIZE;", id="optimize_without_table"),
        pytest.param("OPTIMIZE events WHERE;", id="optimize_where_without_predicate"),
        pytest.param("OPTIMIZE events FULL WHERE;", id="optimize_full_where_without_predicate"),
        pytest.param("OPTIMIZE events ZORDER BY ();", id="optimize_empty_zorder_list"),
        pytest.param("OPTIMIZE events ZORDER BY (a, );", id="optimize_zorder_trailing_comma"),
        pytest.param("VACUUM;", id="vacuum_without_table"),
        pytest.param("VACUUM t FULL LITE;", id="vacuum_full_and_lite"),
        pytest.param("VACUUM t DRY RUN FULL;", id="vacuum_dry_run_and_full"),
    ],
)
def test_maintenance_full_mode_rejections(sql: str) -> None:
    """OPTIMIZE / VACUUM clause boundaries and exclusive FULL/LITE modes."""
    assert _violations(sql), f"Expected violations but got none for:\n{sql}"

@pytest.mark.parametrize(
    "sql",
    [
        pytest.param(
            "MERGE WITH SCHEMA EVOLUTION t USING s ON t.k = s.k WHEN MATCHED THEN UPDATE SET *;",
            id="schema_evolution_without_into",
        ),
    ],
)
def test_merge_schema_evolution_rejections(sql: str) -> None:
    """WITH SCHEMA EVOLUTION is only valid before INTO."""
    assert _violations(sql), f"Expected violations but got none for:\n{sql}"

@pytest.mark.parametrize(
    "sql",
    [
        pytest.param("ALTER CATALOG c DEFAULT COLLATION;", id="catalog_collation_without_value"),
        pytest.param("ALTER CATALOG c SET TAGS ();", id="catalog_empty_tags"),
        pytest.param("ALTER CATALOG c SET MANAGED LOCATION;", id="catalog_managed_location_without_path"),
        pytest.param("ALTER CATALOG c RETAIN DROPPED TO 1;", id="catalog_retain_dropped_without_unit"),
        pytest.param("ALTER SCHEMA s SET DBPROPERTIES ();", id="schema_empty_dbproperties"),
        pytest.param("ALTER SCHEMA s DEFAULT COLLATION;", id="schema_collation_without_value"),
    ],
)
def test_alter_catalog_schema_rejections(sql: str) -> None:
    """ALTER CATALOG / SCHEMA clause boundaries."""
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

@pytest.mark.parametrize(
    "sql",
    [
        pytest.param("ALTER SHARE s ADD TABLE;", id="share_add_table_without_name"),
        pytest.param("ALTER SHARE s ADD;", id="share_add_without_object"),
        pytest.param("ALTER SHARE s RENAME TO;", id="share_rename_without_value"),
    ],
)
def test_alter_share_rejections(sql: str) -> None:
    """ALTER SHARE clause boundaries."""
    assert _violations(sql), f"Expected violations but got none for:\n{sql}"

@pytest.mark.parametrize(
    "sql",
    [
        pytest.param("ALTER RECIPIENT r SET PROPERTIES ();", id="recipient_empty_properties"),
        pytest.param("ALTER RECIPIENT r UNSET PROPERTIES ();", id="recipient_empty_unset_properties"),
        pytest.param("ALTER PROVIDER p RENAME TO;", id="provider_rename_without_value"),
    ],
)
def test_alter_recipient_provider_rejections(sql: str) -> None:
    """ALTER RECIPIENT / PROVIDER boundaries."""
    assert _violations(sql), f"Expected violations but got none for:\n{sql}"

@pytest.mark.parametrize(
    "sql",
    [
        pytest.param("ALTER CONNECTION c RENAME TO;", id="connection_rename_without_value"),
        pytest.param("ALTER CONNECTION c OPTIONS ();", id="connection_empty_options"),
        pytest.param("ALTER EXTERNAL LOCATION l SET URL;", id="location_set_url_without_value"),
        pytest.param("ALTER CREDENTIAL c RENAME TO;", id="credential_rename_without_value"),
    ],
)
def test_alter_connection_location_credential_rejections(sql: str) -> None:
    """ALTER CONNECTION / EXTERNAL LOCATION / CREDENTIAL boundaries."""
    assert _violations(sql), f"Expected violations but got none for:\n{sql}"

@pytest.mark.parametrize(
    "sql",
    [
        pytest.param("ALTER MATERIALIZED VIEW v ALTER COLUMN c COMMENT;", id="mv_column_comment_without_value"),
        pytest.param("ALTER MATERIALIZED VIEW v ADD SCHEDULE;", id="mv_add_schedule_without_clause"),
        pytest.param("ALTER STREAMING TABLE t SET OWNER TO;", id="streaming_table_owner_without_value"),
    ],
)
def test_alter_materialized_view_streaming_table_rejections(sql: str) -> None:
    """ALTER MATERIALIZED VIEW / STREAMING TABLE boundaries."""
    assert _violations(sql), f"Expected violations but got none for:\n{sql}"

@pytest.mark.parametrize(
    "sql",
    [
        pytest.param("ALTER GROUP ADD GROUP h;", id="group_without_principal"),
        pytest.param("ALTER GROUP g ADD;", id="group_add_without_members"),
    ],
)
def test_alter_group_rejections(sql: str) -> None:
    """ALTER GROUP boundaries."""
    assert _violations(sql), f"Expected violations but got none for:\n{sql}"


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param("BEGIN ATOMIC END;", id="block_without_body"),
        pytest.param("BEGIN IF 1 < 2 THEN END IF; END", id="if_without_body"),
        pytest.param("BEGIN WHILE num < 10 SET num = num + 1; END WHILE; END", id="while_without_do"),
        pytest.param("BEGIN LOOP LEAVE; END LOOP; END", id="leave_without_label"),
        pytest.param("BEGIN GET DIAGNOSTICS rc = ; END", id="diagnostics_without_item"),
    ],
)
def test_scripting_rejections(sql: str) -> None:
    """SQL scripting boundaries."""
    assert _violations(sql), f"Expected violations but got none for:\n{sql}"


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param("FSCK REPAIR TABLE;", id="fsck_without_table"),
        pytest.param("FSCK REPAIR TABLE t METADATA;", id="fsck_metadata_without_only"),
        pytest.param("REORG TABLE events;", id="reorg_without_apply"),
        pytest.param("REORG TABLE events APPLY ();", id="reorg_empty_purge"),
        pytest.param("CACHE SELECT FROM boxes;", id="cache_select_without_column"),
        pytest.param("CACHE SELECT a, FROM boxes;", id="cache_select_trailing_comma"),
        pytest.param("DROP BLOOMFILTER INDEX;", id="bloom_without_table"),
        pytest.param("DROP BLOOMFILTER INDEX ON TABLE t FOR COLUMNS ();", id="bloom_empty_columns"),
        pytest.param("REPAIR TABLE;", id="repair_without_table"),
        pytest.param("REFRESH MATERIALIZED VIEW;", id="refresh_mv_without_table"),
        pytest.param("REFRESH FOREIGN;", id="refresh_foreign_without_type"),
        pytest.param("REFRESH FUNCTION;", id="refresh_function_without_name"),
        pytest.param("UNDROP TABLE;", id="undrop_without_name"),
        pytest.param("SYNC TABLE main.t FROM;", id="sync_without_source"),
        pytest.param("LIST;", id="list_without_url"),
        pytest.param("CALL (1);", id="call_without_name"),
        pytest.param("SET RECIPIENT;", id="set_recipient_without_name"),
        pytest.param("ANALYZE TABLE COMPUTE STORAGE METRICS;", id="analyze_metrics_without_table"),
        pytest.param("SET TAG ON TABLE t;", id="set_tag_without_key"),
        pytest.param("UNSET TAG ON TABLE t;", id="unset_tag_without_key"),
    ],
)
def test_maintenance_and_utility_rejections(sql: str) -> None:
    """Delta maintenance and auxiliary statement boundaries."""
    assert _violations(sql), f"Expected violations but got none for:\n{sql}"


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param("DESCRIBE CATALOG;", id="describe_catalog_without_name"),
        pytest.param("DESCRIBE CONNECTION;", id="describe_connection_without_name"),
        pytest.param("DESCRIBE CREDENTIAL;", id="describe_credential_without_name"),
        pytest.param("DESCRIBE EXTERNAL LOCATION;", id="describe_location_without_name"),
        pytest.param("DESCRIBE FUNCTION;", id="describe_function_without_name"),
        pytest.param("DESCRIBE POLICY p;", id="describe_policy_without_target"),
        pytest.param("DESCRIBE PROCEDURE;", id="describe_procedure_without_name"),
        pytest.param("DESCRIBE PROVIDER;", id="describe_provider_without_name"),
        pytest.param("DESCRIBE QUERY;", id="describe_query_without_statement"),
        pytest.param("DESCRIBE SCHEMA;", id="describe_schema_without_name"),
        pytest.param("DESCRIBE SHARE;", id="describe_share_without_name"),
        pytest.param("DESCRIBE VOLUME;", id="describe_volume_without_name"),
        pytest.param("SHOW SHARES IN PROVIDER;", id="show_shares_in_provider_without_name"),
        pytest.param("SHOW ALL IN SHARE;", id="show_all_in_share_without_name"),
        pytest.param("SHOW COLUMNS IN;", id="show_columns_without_table"),
        pytest.param("SHOW GRANTS TO RECIPIENT;", id="show_grants_to_recipient_without_name"),
        pytest.param("SHOW POLICIES ON;", id="show_policies_without_target"),
        pytest.param("DENY SELECT ON TABLE t;", id="deny_without_principal"),
        pytest.param("DROP GROUP;", id="drop_group_without_name"),
        pytest.param("GRANT SELECT ON SHARE s TO RECIPIENT;", id="grant_share_without_recipient"),
        pytest.param("REVOKE SELECT ON SHARE FROM RECIPIENT r;", id="revoke_share_without_share"),
        pytest.param("GRANT SELECT ON TABLE TO `u`;", id="grant_table_without_name"),
    ],
)
def test_uc_show_describe_security_rejections(sql: str) -> None:
    """Unity Catalog SHOW/DESCRIBE and security statement boundaries."""
    assert _violations(sql), f"Expected violations but got none for:\n{sql}"


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param("DROP CONNECTION;", id="drop_connection_without_name"),
        pytest.param("DROP CREDENTIAL;", id="drop_credential_without_name"),
        pytest.param("DROP EXTERNAL LOCATION;", id="drop_location_without_name"),
        pytest.param("DROP POLICY p;", id="drop_policy_without_target"),
        pytest.param("DROP PROCEDURE;", id="drop_procedure_without_name"),
        pytest.param("DROP PROVIDER;", id="drop_provider_without_name"),
        pytest.param("DROP RECIPIENT;", id="drop_recipient_without_name"),
        pytest.param("DROP SHARE;", id="drop_share_without_name"),
        pytest.param("DROP TEMPORARY VARIABLE;", id="drop_variable_without_name"),
        pytest.param("DROP TABLE;", id="drop_table_without_name"),
    ],
)
def test_drop_uc_rejections(sql: str) -> None:
    """Unity Catalog DROP statement boundaries."""
    assert _violations(sql), f"Expected violations but got none for:\n{sql}"


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param("INSERT INTO t REPLACE ON SELECT a FROM s;", id="insert_replace_on_without_expression"),
        pytest.param("RESTORE TABLE employee;", id="restore_without_version"),
        pytest.param("RESTORE TABLE employee TO;", id="restore_without_time_travel"),
        pytest.param("RESTORE TABLE employee TO TIMESTAMP AS OF;", id="restore_timestamp_without_expression"),
        pytest.param("RESTORE TO VERSION AS OF 1;", id="restore_without_table_name"),
    ],
)
def test_insert_and_restore_rejections(sql: str) -> None:
    """INSERT/REPLACE ON and RESTORE boundaries."""
    assert _violations(sql), f"Expected violations but got none for:\n{sql}"


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param("CREATE POLICY p ON CATALOG c;", id="policy_without_body"),
        pytest.param("CREATE PROCEDURE p;", id="procedure_without_body"),
        pytest.param("CREATE TABLE t FLOW INSERT BY NAME;", id="table_flow_without_query"),
    ],
)
def test_policy_procedure_table_flow_rejections(sql: str) -> None:
    """CREATE POLICY / PROCEDURE / pipeline TABLE FLOW boundaries."""
    assert _violations(sql), f"Expected violations but got none for:\n{sql}"


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param("SELECT * FROM t OFFSET;", id="offset_without_expression"),
        pytest.param("SELECT * FROM test TABLESAMPLE ();", id="tablesample_without_sample"),
        pytest.param("SELECT * FROM test TABLESAMPLE (30 PERCENT) REPEATABLE ();", id="repeatable_without_seed"),
        pytest.param("SELECT * FROM t MATCH_RECOGNIZE (DEFINE a AS TRUE);", id="match_recognize_without_pattern"),
        pytest.param("SELECT * FROM t WITH();", id="table_options_empty"),
        pytest.param("WITH RECURSIVE r(n) MAX RECURSION LEVEL AS (VALUES (1)) SELECT * FROM r;", id="cte_recursion_without_level"),
    ],
)
def test_query_surface_rejections(sql: str) -> None:
    """Query-surface boundaries: OFFSET, sampling, MATCH_RECOGNIZE, pipeline."""
    assert _violations(sql), f"Expected violations but got none for:\n{sql}"


@pytest.mark.parametrize(
    "sql",
    [
        pytest.param("SELECT ALL DISTINCT a FROM t;", id="select_all_and_distinct"),
        pytest.param("SELECT a FROM t GROUP BY ALL, a;", id="group_by_all_and_list"),
        pytest.param("SELECT * FROM t ORDER BY ALL, a;", id="order_by_all_and_list"),
        pytest.param("USE SCHEMA;", id="use_schema_without_name"),
        pytest.param("CREATE TEMP EXTERNAL TABLE t (a INT);", id="create_table_temp_external"),
        pytest.param(
            "CREATE OR REPLACE TEMP TABLE IF NOT EXISTS t (a INT);",
            id="create_table_replace_and_if_not_exists",
        ),
        pytest.param(
            "CREATE TABLE t (CONSTRAINT pk PRIMARY KEY (a));",
            id="create_table_constraint_without_column",
        ),
    ],
)
def test_rejection_hardening(sql: str) -> None:
    """Over-acceptances tightened to the reference."""
    assert _violations(sql), f"Expected violations but got none for:\n{sql}"
