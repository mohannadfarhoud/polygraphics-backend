from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS model_try_on_config (
  job_id TEXT PRIMARY KEY NOT NULL,
  hanger_x REAL NOT NULL,
  hanger_y REAL NOT NULL,
  hanger_z REAL NOT NULL,
  profile_x REAL,
  profile_y REAL,
  profile_z REAL,
  spin_deg REAL NOT NULL DEFAULT 0,
  pitch_deg REAL NOT NULL DEFAULT 0,
  roll_deg REAL NOT NULL DEFAULT 0,
  api_vertical_flip INTEGER NOT NULL DEFAULT 0,
  offset_vertical REAL NOT NULL DEFAULT 0,
  offset_depth REAL NOT NULL DEFAULT 0,
  offset_lateral REAL NOT NULL DEFAULT 0,
  nudge_left REAL NOT NULL DEFAULT 0,
  nudge_right REAL NOT NULL DEFAULT 0,
  jewelry_type TEXT NOT NULL DEFAULT 'drop',
  config_version INTEGER NOT NULL DEFAULT 1,
  owner_user_id TEXT,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS user_try_on_calibration (
  user_id TEXT PRIMARY KEY NOT NULL,
  left_vertical REAL NOT NULL DEFAULT 0,
  left_depth REAL NOT NULL DEFAULT 0,
  left_lateral REAL NOT NULL DEFAULT 0,
  right_vertical REAL NOT NULL DEFAULT 0,
  right_depth REAL NOT NULL DEFAULT 0,
  right_lateral REAL NOT NULL DEFAULT 0,
  updated_at REAL NOT NULL
);
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


def get_model_try_on(db_path: Path, job_id: str) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM model_try_on_config WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if not row:
            return None
        return dict(row)


def upsert_model_try_on(db_path: Path, row: dict[str, Any]) -> dict[str, Any]:
    now = float(time.time())
    with _connect(db_path) as conn:
        existing = conn.execute(
            "SELECT owner_user_id FROM model_try_on_config WHERE job_id = ?",
            (row["job_id"],),
        ).fetchone()
        owner = row.get("owner_user_id")
        if existing and existing["owner_user_id"]:
            owner = existing["owner_user_id"]
        elif owner is None and existing:
            owner = existing["owner_user_id"]
        conn.execute(
            """
            INSERT OR REPLACE INTO model_try_on_config (
              job_id, hanger_x, hanger_y, hanger_z,
              profile_x, profile_y, profile_z,
              spin_deg, pitch_deg, roll_deg,
              api_vertical_flip,
              offset_vertical, offset_depth, offset_lateral,
              nudge_left, nudge_right,
              jewelry_type, config_version, owner_user_id, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["job_id"],
                row["hanger_x"],
                row["hanger_y"],
                row["hanger_z"],
                row.get("profile_x"),
                row.get("profile_y"),
                row.get("profile_z"),
                row["spin_deg"],
                row["pitch_deg"],
                row["roll_deg"],
                int(bool(row["api_vertical_flip"])),
                row["offset_vertical"],
                row["offset_depth"],
                row["offset_lateral"],
                row["nudge_left"],
                row["nudge_right"],
                row["jewelry_type"],
                int(row.get("config_version") or 1),
                owner,
                now,
            ),
        )
        conn.commit()
    saved = get_model_try_on(db_path, row["job_id"])
    assert saved is not None
    return saved


def delete_model_try_on(db_path: Path, job_id: str) -> None:
    with _connect(db_path) as conn:
        conn.execute("DELETE FROM model_try_on_config WHERE job_id = ?", (job_id,))
        conn.commit()


def get_user_calibration(db_path: Path, user_id: str) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM user_try_on_calibration WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if not row:
            return None
        return dict(row)


def upsert_user_calibration(db_path: Path, user_id: str, row: dict[str, Any]) -> dict[str, Any]:
    now = float(time.time())
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO user_try_on_calibration (
              user_id,
              left_vertical, left_depth, left_lateral,
              right_vertical, right_depth, right_lateral,
              updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                row["left_vertical"],
                row["left_depth"],
                row["left_lateral"],
                row["right_vertical"],
                row["right_depth"],
                row["right_lateral"],
                now,
            ),
        )
        conn.commit()
    saved = get_user_calibration(db_path, user_id)
    assert saved is not None
    return saved
