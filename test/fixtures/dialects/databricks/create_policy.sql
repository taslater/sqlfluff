-- CREATE POLICY. https://docs.databricks.com/aws/en/sql/language-manual/sql-ref-syntax-ddl-create-policy
CREATE POLICY ssn_mask
ON CATALOG employees
COLUMN MASK ssn_to_last_nr
TO 'All Users' EXCEPT 'HR admins'
FOR TABLES
MATCH COLUMNS has_tag('ssn') AS ssn
ON COLUMN ssn
USING COLUMNS (4);

CREATE POLICY model_access
ON SCHEMA system.ai
TO data_scientists EXCEPT contractors
GRANT EXECUTE FOR MODEL SERVICES
WHEN has_tag_value('ai.model_creator', 'anthropic');
