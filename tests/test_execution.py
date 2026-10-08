import sqlite3
from pathlib import Path

from agentic_rl_sql.execution import execute_sql, execution_match, result_sets_equal


def make_db(path: Path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE t(id INTEGER, name TEXT);
        INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, 'b');
        """
    )
    conn.commit()
    conn.close()


def test_execute_and_semantic_match(tmp_path):
    db = tmp_path / "x.sqlite"
    make_db(db)
    ok, pred, gold = execution_match(db, "SELECT name FROM t WHERE id > 1", "SELECT name FROM t WHERE id IN (2,3)")
    assert ok
    assert pred.executed and gold.executed


def test_unordered_results_compare_as_multiset(tmp_path):
    db = tmp_path / "x.sqlite"
    make_db(db)
    a = execute_sql(db, "SELECT name FROM t")
    b = execute_sql(db, "SELECT name FROM t ORDER BY id DESC")
    assert not result_sets_equal(a, b)
    c = execute_sql(db, "SELECT name FROM t WHERE id IN (3,2,1)")
    assert result_sets_equal(a, c)


def test_database_is_read_only(tmp_path):
    db = tmp_path / "x.sqlite"
    make_db(db)
    result = execute_sql(db, "DROP TABLE t")
    assert not result.safe
    conn = sqlite3.connect(db)
    assert conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 3
    conn.close()
