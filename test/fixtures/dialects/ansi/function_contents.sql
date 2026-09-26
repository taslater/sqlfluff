-- The named function-argument forms the grammar spells out. Function
-- arguments are comma-delimited; each of these is one primary form, not a
-- sequence of adjacent expressions.
SELECT
    substring(s FROM 2 FOR 3),
    substring(s FROM 2),
    extract(YEAR FROM d),
    trim(BOTH ' ' FROM s),
    position('a' IN 'ab'),
    count(*),
    count(DISTINCT a, b),
    cast(a AS INT),
    first(age IGNORE NULLS),
    string_agg(x, ',' ORDER BY y)
FROM t;
