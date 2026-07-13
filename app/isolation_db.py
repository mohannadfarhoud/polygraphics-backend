from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS isolation_datasets (
  dataset_id TEXT PRIMARY KEY NOT NULL,
  name TEXT NOT NULL,
  owner_user_id TEXT,
  pair_count INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_isolation_datasets_owner ON isolation_datasets(owner_user_id);

CREATE TABLE IF NOT EXISTS isolation_train_jobs (
  job_id TEXT PRIMARY KEY NOT NULL,
  dataset_id TEXT NOT NULL,
  status TEXT NOT NULL,
  progress INTEGER NOT NULL DEFAULT 0,
  base_model TEXT,
  epochs INTEGER,
  val_split REAL,
  model_id TEXT,
  metrics_json TEXT,
  error TEXT,
  provider TEXT,
  owner_user_id TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_isolation_train_status ON isolation_train_jobs(status);

CREATE TABLE IF NOT EXISTS isolation_models (
  model_id TEXT PRIMARY KEY NOT NULL,
  name TEXT NOT NULL,
  dataset_id TEXT,
  onnx_rel_path TEXT NOT NULL,
  metrics_json TEXT,
  is_active INTEGER NOT NULL DEFAULT 0,
  owner_user_id TEXT,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_isolation_models_active ON isolation_models(is_active);
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


def insert_dataset(db_path: Path, *, name: str, owner_user_id: str | None) -> dict[str, Any]:
    now = float(time.time())
    dataset_id = str(uuid.uuid4())
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO isolation_datasets (dataset_id, name, owner_user_id, pair_count, created_at, updated_at)
            VALUES (?, ?, ?, 0, ?, ?)
            """,
            (dataset_id, name, owner_user_id, now, now),
        )
        conn.commit()
    row = get_dataset(db_path, dataset_id)
    assert row is not None
    return row


def list_datasets(db_path: Path) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM isolation_datasets ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_dataset(db_path: Path, dataset_id: str) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM isolation_datasets WHERE dataset_id = ?",
            (dataset_id,),
        ).fetchone()
    return dict(row) if row else None


def update_dataset(db_path: Path, dataset_id: str, **fields: Any) -> dict[str, Any] | None:
    if not fields:
        return get_dataset(db_path, dataset_id)
    fields["updated_at"] = float(time.time())
    cols = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [dataset_id]
    with _connect(db_path) as conn:
        conn.execute(f"UPDATE isolation_datasets SET {cols} WHERE dataset_id = ?", vals)
        conn.commit()
    return get_dataset(db_path, dataset_id)


def delete_dataset(db_path: Path, dataset_id: str) -> bool:
    with _connect(db_path) as conn:
        cur = conn.execute("DELETE FROM isolation_datasets WHERE dataset_id = ?", (dataset_id,))
        conn.commit()
    return cur.rowcount > 0


def insert_train_job(db_path: Path, row: dict[str, Any]) -> dict[str, Any]:
    now = float(time.time())
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO isolation_train_jobs (
              job_id, dataset_id, status, progress, base_model, epochs, val_split,
              model_id, metrics_json, error, provider, owner_user_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["job_id"],
                row["dataset_id"],
                row["status"],
                int(row.get("progress") or 0),
                row.get("base_model"),
                row.get("epochs"),
                row.get("val_split"),
                row.get("model_id"),
                row.get("metrics_json"),
                row.get("error"),
                row.get("provider"),
                row.get("owner_user_id"),
                now,
                now,
            ),
        )
        conn.commit()
    saved = get_train_job(db_path, row["job_id"])
    assert saved is not None
    return saved


def get_train_job(db_path: Path, job_id: str) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM isolation_train_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
    return dict(row) if row else None


def update_train_job(db_path: Path, job_id: str, **fields: Any) -> dict[str, Any] | None:
    if not fields:
        return get_train_job(db_path, job_id)
    fields["updated_at"] = float(time.time())
    cols = ", ".join(f"{k} = ?" for k in fields)
    vals = list(fields.values()) + [job_id]
    with _connect(db_path) as conn:
        conn.execute(f"UPDATE isolation_train_jobs SET {cols} WHERE job_id = ?", vals)
        conn.commit()
    return get_train_job(db_path, job_id)


def claim_next_train_job(db_path: Path) -> dict[str, Any] | None:
    now = float(time.time())
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM isolation_train_jobs
            WHERE status = ? AND (provider IS NULL OR provider = ?)
            ORDER BY created_at ASC LIMIT 1
            """,
            ("queued", "worker_gpu"),
        ).fetchone()
        if not row:
            return None
        job_id = row["job_id"]
        conn.execute(
            """
            UPDATE isolation_train_jobs SET status = ?, progress = ?, updated_at = ?
            WHERE job_id = ? AND status = ?
            """,
            ("running", 5, now, job_id, "queued"),
        )
        if conn.total_changes == 0:
            return None
        conn.commit()
        claimed = conn.execute(
            "SELECT * FROM isolation_train_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        return dict(claimed) if claimed else None


def insert_model(db_path: Path, row: dict[str, Any]) -> dict[str, Any]:
    now = float(time.time())
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO isolation_models (
              model_id, name, dataset_id, onnx_rel_path, metrics_json, is_active, owner_user_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["model_id"],
                row["name"],
                row.get("dataset_id"),
                row["onnx_rel_path"],
                row.get("metrics_json"),
                int(row.get("is_active") or 0),
                row.get("owner_user_id"),
                now,
            ),
        )
        conn.commit()
    saved = get_model(db_path, row["model_id"])
    assert saved is not None
    return saved


def list_models(db_path: Path) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM isolation_models ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_model(db_path: Path, model_id: str) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM isolation_models WHERE model_id = ?",
            (model_id,),
        ).fetchone()
    return dict(row) if row else None


def get_active_model(db_path: Path) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM isolation_models WHERE is_active = 1 ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def set_active_model(db_path: Path, model_id: str) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        conn.execute("UPDATE isolation_models SET is_active = 0")
        conn.execute(
            "UPDATE isolation_models SET is_active = 1 WHERE model_id = ?",
            (model_id,),
        )
        conn.commit()
    return get_model(db_path, model_id)


def metrics_from_json(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None
