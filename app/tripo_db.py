from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS tripo_jobs (
  tripo_job_id TEXT PRIMARY KEY NOT NULL,
  user_id TEXT NOT NULL,
  status TEXT NOT NULL,
  input_image_path TEXT NOT NULL,
  model_path TEXT,
  tripo_task_id TEXT,
  provider TEXT,
  error TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tripo_jobs_user ON tripo_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_tripo_jobs_status ON tripo_jobs(status);
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
        try:
            conn.execute("ALTER TABLE tripo_jobs ADD COLUMN provider TEXT")
        except sqlite3.OperationalError:
            pass
        conn.commit()


def insert_job(db_path: Path, row: dict[str, Any]) -> dict[str, Any]:
    now = float(time.time())
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO tripo_jobs (
              tripo_job_id, user_id, status, input_image_path,
              model_path, tripo_task_id, provider, error, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["tripo_job_id"],
                row["user_id"],
                row["status"],
                row["input_image_path"],
                row.get("model_path"),
                row.get("tripo_task_id"),
                row.get("provider"),
                row.get("error"),
                now,
                now,
            ),
        )
        conn.commit()
    saved = get_job(db_path, row["tripo_job_id"])
    assert saved is not None
    return saved


def get_job(db_path: Path, tripo_job_id: str) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM tripo_jobs WHERE tripo_job_id = ?",
            (tripo_job_id,),
        ).fetchone()
        if not row:
            return None
        return dict(row)


def update_job(db_path: Path, tripo_job_id: str, **fields: Any) -> dict[str, Any] | None:
    if not fields:
        return get_job(db_path, tripo_job_id)
    fields["updated_at"] = float(time.time())
    cols = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [tripo_job_id]
    with _connect(db_path) as conn:
        conn.execute(
            f"UPDATE tripo_jobs SET {cols} WHERE tripo_job_id = ?",
            vals,
        )
        conn.commit()
    return get_job(db_path, tripo_job_id)


def claim_next_worker_job(db_path: Path) -> dict[str, Any] | None:
    """Atomically claim the oldest queued Tripo job for GPU worker processing."""
    now = float(time.time())
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM tripo_jobs
            WHERE status = ? AND (provider IS NULL OR provider = ?)
            ORDER BY created_at ASC
            LIMIT 1
            """,
            ("queued", "worker_triposr"),
        ).fetchone()
        if not row:
            return None
        tripo_job_id = row["tripo_job_id"]
        conn.execute(
            """
            UPDATE tripo_jobs
            SET status = ?, updated_at = ?
            WHERE tripo_job_id = ? AND status = ?
            """,
            ("processing", now, tripo_job_id, "queued"),
        )
        if conn.total_changes == 0:
            return None
        conn.commit()
        claimed = conn.execute(
            "SELECT * FROM tripo_jobs WHERE tripo_job_id = ?",
            (tripo_job_id,),
        ).fetchone()
        return dict(claimed) if claimed else None


def claim_next_worker_job(db_path: Path) -> dict[str, Any] | None:
    """Atomically claim the oldest queued Tripo job for GPU worker processing."""
    now = float(time.time())
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM tripo_jobs
            WHERE status = ? AND (provider IS NULL OR provider = ?)
            ORDER BY created_at ASC
            LIMIT 1
            """,
            ("queued", "worker_triposr"),
        ).fetchone()
        if not row:
            return None
        tripo_job_id = row["tripo_job_id"]
        conn.execute(
            """
            UPDATE tripo_jobs
            SET status = ?, updated_at = ?
            WHERE tripo_job_id = ? AND status = ?
            """,
            ("processing", now, tripo_job_id, "queued"),
        )
        if conn.total_changes == 0:
            return None
        conn.commit()
        claimed = conn.execute(
            "SELECT * FROM tripo_jobs WHERE tripo_job_id = ?",
            (tripo_job_id,),
        ).fetchone()
        return dict(claimed) if claimed else None
