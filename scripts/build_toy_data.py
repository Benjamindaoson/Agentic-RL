#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agentic_rl_sql.dataset import write_parquet
from agentic_rl_sql.types import SqlTask


def build_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE departments (department_id INTEGER PRIMARY KEY, department_name TEXT NOT NULL);
        CREATE TABLE employees (
            employee_id INTEGER PRIMARY KEY,
            employee_name TEXT NOT NULL,
            department_id INTEGER NOT NULL,
            salary INTEGER NOT NULL,
            FOREIGN KEY(department_id) REFERENCES departments(department_id)
        );
        INSERT INTO departments VALUES (1, 'Engineering'), (2, 'Sales');
        INSERT INTO employees VALUES
            (1, 'Alice', 1, 120000), (2, 'Bob', 1, 100000),
            (3, 'Carol', 2, 90000), (4, 'Dan', 2, 95000);
    """)
    conn.commit()
    conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="data/toy")
    args = ap.parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    db = out / "company.sqlite"
    build_db(db)
    train = [
        SqlTask("toy-train-1", "toy", "train", "company", str(db), "Which department has the highest average salary? Return the department name.", "SELECT d.department_name FROM departments d JOIN employees e ON d.department_id=e.department_id GROUP BY d.department_id ORDER BY AVG(e.salary) DESC LIMIT 1"),
        SqlTask("toy-train-2", "toy", "train", "company", str(db), "How many employees work in Sales?", "SELECT COUNT(*) FROM employees e JOIN departments d ON e.department_id=d.department_id WHERE d.department_name='Sales'"),
    ]
    val = [SqlTask("toy-val-1", "toy", "val", "company", str(db), "List employee names in descending salary order.", "SELECT employee_name FROM employees ORDER BY salary DESC")]
    write_parquet(train, out / "train.parquet")
    write_parquet(val, out / "val.parquet")
    with (out / "tasks.jsonl").open("w", encoding="utf-8") as f:
        for task in train + val:
            f.write(json.dumps(task.to_dict(), ensure_ascii=False) + "\n")
    print(json.dumps({"database": str(db), "train": len(train), "val": len(val)}, indent=2))


if __name__ == "__main__":
    main()
