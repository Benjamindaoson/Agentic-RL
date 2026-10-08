from __future__ import annotations

from dataclasses import dataclass
import re

import sqlglot
from sqlglot import exp


@dataclass(slots=True)
class GuardResult:
    valid: bool
    safe: bool
    normalized_sql: str | None
    error: str | None
    order_sensitive: bool = False


_BLOCKED_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Create,
    exp.Drop,
    exp.Alter,
    exp.Command,
    exp.Transaction,
    exp.Merge,
)

_BLOCKED_WORDS = {
    "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REINDEX", "ANALYZE",
    "LOAD_EXTENSION", "INSTALL", "COPY",
}


def _is_read_query(tree: exp.Expression) -> bool:
    if isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.Subquery)):
        return True
    return tree.find(exp.Select) is not None and not isinstance(tree, _BLOCKED_NODES)


def validate_read_only_sql(sql: str, dialect: str = "sqlite") -> GuardResult:
    text = (sql or "").strip().rstrip(";").strip()
    if not text:
        return GuardResult(False, False, None, "empty SQL")
    upper = text.upper()
    if any(re.search(rf"\b{re.escape(word)}\b", upper) for word in _BLOCKED_WORDS):
        return GuardResult(True, False, None, "blocked SQL operation")
    try:
        trees = sqlglot.parse(text, read=dialect)
    except Exception as exc:
        return GuardResult(False, False, None, f"parse error: {exc}")
    if len(trees) != 1:
        return GuardResult(False, False, None, "exactly one SQL statement is required")
    tree = trees[0]
    if tree is None:
        return GuardResult(False, False, None, "empty parse tree")
    if isinstance(tree, _BLOCKED_NODES) or any(tree.find(node) is not None for node in _BLOCKED_NODES):
        return GuardResult(True, False, None, "write or DDL statement is forbidden")
    if not _is_read_query(tree):
        return GuardResult(True, False, None, "only SELECT/WITH queries are allowed")
    normalized = tree.sql(dialect=dialect, pretty=False)
    return GuardResult(True, True, normalized, None, tree.find(exp.Order) is not None)
