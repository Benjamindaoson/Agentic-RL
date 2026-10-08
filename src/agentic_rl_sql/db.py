from __future__ import annotations

import sqlite3
from pathlib import Path


def connect_read_only(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    uri = f"file:{path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA trusted_schema=OFF")
    return conn


def render_schema(db_path: str | Path, max_chars: int = 12000, sample_values: int = 0) -> str:
    conn = connect_read_only(db_path)
    try:
        tables = [row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()]
        chunks: list[str] = []
        for table in tables:
            escaped = table.replace('"', '""')
            columns = conn.execute(f'PRAGMA table_info("{escaped}")').fetchall()
            foreign_keys = conn.execute(f'PRAGMA foreign_key_list("{escaped}")').fetchall()
            col_text = ", ".join(f"{row[1]} {row[2] or 'TEXT'}{' PRIMARY KEY' if row[5] else ''}" for row in columns)
            chunks.append(f"TABLE {table} ({col_text})")
            for fk in foreign_keys:
                chunks.append(f"  FK {table}.{fk[3]} -> {fk[2]}.{fk[4]}")
            if sample_values > 0:
                try:
                    rows = conn.execute(f'SELECT * FROM "{escaped}" LIMIT ?', (sample_values,)).fetchall()
                    if rows:
                        chunks.append(f"  SAMPLE {rows}")
                except sqlite3.Error:
                    pass
            rendered = "\n".join(chunks)
            if len(rendered) >= max_chars:
                return rendered[:max_chars] + "\n... [schema truncated]"
        return "\n".join(chunks)
    finally:
        conn.close()
