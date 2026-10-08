import sqlite3

import pytest

from agentic_rl_sql.db import connect_read_only
from agentic_rl_sql.execution import execute_sql


def test_readonly_sqlite_handles_uri_special_characters(tmp_path):
    path = tmp_path / "question ? # & percent%.sqlite"
    with sqlite3.connect(path) as conn:
        conn.executescript("CREATE TABLE sample(x INTEGER); INSERT INTO sample VALUES (17);")
    with connect_read_only(path) as conn:
        assert conn.execute("SELECT x FROM sample").fetchone()[0] == 17
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute("DELETE FROM sample")
    assert execute_sql(path, "SELECT x FROM sample").executed


def test_sql_budget_rejects_invalid_values(tmp_path):
    path = tmp_path / "tiny.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE sample(x INTEGER)")
    with pytest.raises(ValueError):
        execute_sql(path, "SELECT * FROM sample", timeout_seconds=0)
    with pytest.raises(ValueError):
        execute_sql(path, "SELECT * FROM sample", max_rows=0)
