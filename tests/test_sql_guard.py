from agentic_rl_sql.sql_guard import validate_read_only_sql


def test_select_is_allowed():
    result = validate_read_only_sql("SELECT name FROM users WHERE id = 1")
    assert result.valid and result.safe
    assert result.normalized_sql


def test_with_query_is_allowed():
    result = validate_read_only_sql("WITH x AS (SELECT 1 AS a) SELECT a FROM x")
    assert result.valid and result.safe


def test_write_and_multiple_statements_are_blocked():
    assert not validate_read_only_sql("DELETE FROM users").safe
    multi = validate_read_only_sql("SELECT 1; SELECT 2")
    assert not multi.valid


def test_pragma_and_attach_are_blocked():
    assert not validate_read_only_sql("PRAGMA table_info(users)").safe
    assert not validate_read_only_sql("ATTACH DATABASE 'x.db' AS x").safe
