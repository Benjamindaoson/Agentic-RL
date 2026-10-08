from __future__ import annotations

import math
import sqlite3
import time
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .db import connect_read_only
from .sql_guard import validate_read_only_sql
from .types import ExecutionResult


def _normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, Decimal)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return str(value)
        try:
            return str(Decimal(str(value)).normalize())
        except InvalidOperation:
            return str(value)
    return " ".join(str(value).strip().split())


def normalize_rows(rows: list[tuple[Any, ...]] | list[list[Any]]) -> list[tuple[Any, ...]]:
    return [tuple(_normalize_scalar(v) for v in row) for row in rows]


def execute_sql(db_path: str | Path, sql: str, timeout_seconds: float = 8.0, max_rows: int = 100000) -> ExecutionResult:
    guard = validate_read_only_sql(sql)
    if not guard.valid:
        return ExecutionResult(sql=sql, valid=False, safe=False, executed=False, error_type="parse_error", error_message=guard.error)
    if not guard.safe or not guard.normalized_sql:
        return ExecutionResult(sql=sql, valid=True, safe=False, executed=False, error_type="unsafe_sql", error_message=guard.error)
    if timeout_seconds <= 0 or max_rows < 1:
        raise ValueError("SQL timeout must be positive and max_rows >= 1")
    conn = connect_read_only(db_path)
    deadline = time.monotonic() + timeout_seconds

    def progress() -> int:
        return 1 if time.monotonic() > deadline else 0

    conn.set_progress_handler(progress, 1000)
    started = time.perf_counter()
    try:
        cur = conn.execute(guard.normalized_sql)
        columns = [str(d[0]) for d in (cur.description or [])]
        raw_rows = cur.fetchmany(max_rows + 1)
        truncated = len(raw_rows) > max_rows
        raw_rows = raw_rows[:max_rows]
        rows = [list(r) for r in normalize_rows(raw_rows)]
        return ExecutionResult(
            sql=guard.normalized_sql,
            valid=True,
            safe=True,
            executed=True,
            rows=rows,
            columns=columns,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            truncated=truncated,
            order_sensitive=guard.order_sensitive,
            row_count=len(rows),
        )
    except sqlite3.OperationalError as exc:
        message = str(exc)
        kind = "timeout" if "interrupted" in message.lower() else "execution_error"
        return ExecutionResult(
            sql=guard.normalized_sql,
            valid=True,
            safe=True,
            executed=False,
            error_type=kind,
            error_message=message,
            elapsed_ms=(time.perf_counter() - started) * 1000,
            order_sensitive=guard.order_sensitive,
        )
    except sqlite3.Error as exc:
        return ExecutionResult(
            sql=guard.normalized_sql,
            valid=True,
            safe=True,
            executed=False,
            error_type="execution_error",
            error_message=str(exc),
            elapsed_ms=(time.perf_counter() - started) * 1000,
            order_sensitive=guard.order_sensitive,
        )
    finally:
        conn.close()


def result_sets_equal(pred: ExecutionResult, gold: ExecutionResult) -> bool:
    if not pred.executed or not gold.executed or pred.truncated or gold.truncated:
        return False
    if len(pred.columns) != len(gold.columns):
        return False
    pred_rows = [tuple(row) for row in pred.rows]
    gold_rows = [tuple(row) for row in gold.rows]
    if gold.order_sensitive:
        return pred_rows == gold_rows
    return Counter(pred_rows) == Counter(gold_rows)


def execution_match(
    db_path: str | Path,
    predicted_sql: str,
    gold_sql: str,
    timeout_seconds: float = 8.0,
    max_rows: int = 100000,
) -> tuple[bool, ExecutionResult, ExecutionResult]:
    pred = execute_sql(db_path, predicted_sql, timeout_seconds=timeout_seconds, max_rows=max_rows)
    gold = execute_sql(db_path, gold_sql, timeout_seconds=timeout_seconds, max_rows=max_rows)
    if not gold.executed:
        raise RuntimeError(f"gold SQL failed for {db_path}: {gold.error_type}: {gold.error_message}")
    return result_sets_equal(pred, gold), pred, gold
