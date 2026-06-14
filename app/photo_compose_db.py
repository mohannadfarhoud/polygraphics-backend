from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS photo_compose_jobs (
  compose_id TEXT PRIMARY KEY NOT NULL,
  job_id TEXT NOT NULL,
  user_id TEXT,
  status TEXT NOT NULL,
  placement_x REAL NOT NULL,
  placement_y REAL NOT NULL,
  image_width INTEGER NOT NULL,
  image_height INTEGER NOT NULL,
  placement_side TEXT NOT NULL DEFAULT 'auto',
  face_image_path TEXT NOT NULL,
  result_url TEXT,
  error TEXT,
  revised_prompt TEXT,
  user_prompt TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_photo_compose_status ON photo_compose_jobs(status);
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


def insert_job(db_path: Path, row: dict[str, Any]) -> dict[str, Any]:
    now = float(time.time())
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO photo_compose_jobs (
              compose_id, job_id, user_id, status,
              placement_x, placement_y, image_width, image_height, placement_side,
              face_image_path, result_url, error, revised_prompt, user_prompt,
              created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["compose_id"],
                row["job_id"],
                row.get("user_id"),
                row["status"],
                row["placement_x"],
                row["placement_y"],
                int(row["image_width"]),
                int(row["image_height"]),
                row.get("placement_side") or "auto",
                row["face_image_path"],
                row.get("result_url"),
                row.get("error"),
                row.get("revised_prompt"),
                row.get("user_prompt"),
                now,
                now,
            ),
        )
        conn.commit()
    saved = get_job(db_path, row["compose_id"])
    assert saved is not None
    return saved


def get_job(db_path: Path, compose_id: str) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM photo_compose_jobs WHERE compose_id = ?",
            (compose_id,),
        ).fetchone()
        if not row:
            return None
        return dict(row)


def update_job(db_path: Path, compose_id: str, **fields: Any) -> dict[str, Any] | None:
    if not fields:
        return get_job(db_path, compose_id)
    fields["updated_at"] = float(time.time())
    cols = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [compose_id]
    with _connect(db_path) as conn:
        conn.execute(
            f"UPDATE photo_compose_jobs SET {cols} WHERE compose_id = ?",
            vals,
        )
        conn.commit()
    return get_job(db_path, compose_id)
