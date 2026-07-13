from __future__ import annotations

import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS user_reconstruction_usage (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT NOT NULL,
  job_id TEXT NOT NULL,
  charged_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_user_reconstruction_usage_user_time
  ON user_reconstruction_usage(user_id, charged_at DESC);
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(db_path: Path) -> None:
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.commit()


def _month_bounds_utc(year: int, month: int) -> tuple[float, float]:
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    return start.timestamp(), end.timestamp()


def current_month_bounds_utc() -> tuple[int, int, float, float]:
    now = datetime.now(timezone.utc)
    start_ts, end_ts = _month_bounds_utc(now.year, now.month)
    return now.year, now.month, start_ts, end_ts


def record_reconstruction_start(db_path: Path, *, user_id: str, job_id: str) -> None:
    now = time.time()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO user_reconstruction_usage (user_id, job_id, charged_at)
            VALUES (?, ?, ?)
            """,
            (user_id, job_id, now),
        )
        conn.commit()


def count_user_starts_in_current_month(db_path: Path, user_id: str) -> int:
    _year, _month, start_ts, end_ts = current_month_bounds_utc()
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) FROM user_reconstruction_usage
            WHERE user_id = ? AND charged_at >= ? AND charged_at < ?
            """,
            (user_id, start_ts, end_ts),
        ).fetchone()
    return int(row[0] if row else 0)


def job_already_charged_this_month(db_path: Path, *, user_id: str, job_id: str) -> bool:
    _year, _month, start_ts, end_ts = current_month_bounds_utc()
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT 1 FROM user_reconstruction_usage
            WHERE user_id = ? AND job_id = ? AND charged_at >= ? AND charged_at < ?
            LIMIT 1
            """,
            (user_id, job_id, start_ts, end_ts),
        ).fetchone()
    return row is not None
