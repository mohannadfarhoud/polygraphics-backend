"""Isolation predict quota (monthly, UTC)."""

from __future__ import annotations

import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from .quota_db import current_month_bounds_utc


class IsolationQuotaStatus(BaseModel):
    used: int
    limit: int
    remaining: int
    period_start: str
    period_end: str


SCHEMA = """
CREATE TABLE IF NOT EXISTS user_isolation_usage (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT NOT NULL,
  predict_id TEXT NOT NULL,
  charged_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_user_isolation_usage_user_time
  ON user_isolation_usage(user_id, charged_at DESC);
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


def free_isolation_per_month() -> int:
    # 0 or negative = unlimited (public / no-auth mode).
    raw = os.getenv("APP_FREE_ISOLATION_PER_MONTH", "0").strip()
    try:
        return int(raw)
    except ValueError:
        return 0


def get_isolation_quota(db_path: Path, user_id: str) -> IsolationQuotaStatus:
    year, month, start_ts, end_ts = current_month_bounds_utc()
    limit = free_isolation_per_month()
    if limit <= 0:
        period_start = datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime("%Y-%m-%d")
        period_end = datetime.fromtimestamp(end_ts, tz=timezone.utc).strftime("%Y-%m-%d")
        return IsolationQuotaStatus(
            used=0,
            limit=0,
            remaining=999999,
            period_start=period_start,
            period_end=period_end,
        )
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) FROM user_isolation_usage
            WHERE user_id = ? AND charged_at >= ? AND charged_at < ?
            """,
            (user_id, start_ts, end_ts),
        ).fetchone()
    used = int(row[0] if row else 0)
    period_start = datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime("%Y-%m-%d")
    period_end = datetime.fromtimestamp(end_ts, tz=timezone.utc).strftime("%Y-%m-%d")
    return IsolationQuotaStatus(
        used=used,
        limit=limit,
        remaining=max(0, limit - used),
        period_start=period_start,
        period_end=period_end,
    )


class IsolationQuotaExceededError(RuntimeError):
    pass


def charge_isolation_predict(db_path: Path, *, user_id: str | None, predict_id: str) -> None:
    """No-op when auth/quota disabled (limit<=0) or user_id is None."""
    if not user_id:
        return
    limit = free_isolation_per_month()
    if limit <= 0:
        return
    quota = get_isolation_quota(db_path, user_id)
    if quota.used >= quota.limit:
        raise IsolationQuotaExceededError(
            f"Monthly isolation predict limit reached ({quota.limit}). Resets on {quota.period_end}."
        )
    now = time.time()
    with _connect(db_path) as conn:
        conn.execute(
            "INSERT INTO user_isolation_usage (user_id, predict_id, charged_at) VALUES (?, ?, ?)",
            (user_id, predict_id, now),
        )
        conn.commit()
