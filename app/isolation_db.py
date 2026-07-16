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

CREATE TABLE IF NOT EXISTS isolation_upload_events (
  event_id TEXT PRIMARY KEY NOT NULL,
  dataset_id TEXT,
  user_id TEXT,
  submitted_photos INTEGER NOT NULL DEFAULT 0,
  successful_photos INTEGER NOT NULL DEFAULT 0,
  unsuccessful_photos INTEGER NOT NULL DEFAULT 0,
  submitted_pairs INTEGER NOT NULL DEFAULT 0,
  successful_pairs INTEGER NOT NULL DEFAULT 0,
  unsuccessful_pairs INTEGER NOT NULL DEFAULT 0,
  error TEXT,
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_isolation_upload_events_dataset
  ON isolation_upload_events(dataset_id);
CREATE INDEX IF NOT EXISTS idx_isolation_upload_events_user
  ON isolation_upload_events(user_id);

CREATE TABLE IF NOT EXISTS isolation_migrations (
  name TEXT PRIMARY KEY NOT NULL,
  applied_at REAL NOT NULL
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
        # Incremental train columns (safe if already present).
        cols = {r[1] for r in conn.execute("PRAGMA table_info(isolation_train_jobs)").fetchall()}
        if "resume_from_model_id" not in cols:
            conn.execute("ALTER TABLE isolation_train_jobs ADD COLUMN resume_from_model_id TEXT")
        if "grow_active" not in cols:
            conn.execute("ALTER TABLE isolation_train_jobs ADD COLUMN grow_active INTEGER NOT NULL DEFAULT 1")
        if "auto_activate" not in cols:
            conn.execute("ALTER TABLE isolation_train_jobs ADD COLUMN auto_activate INTEGER NOT NULL DEFAULT 1")
        # Existing pairs predate upload auditing. Backfill them once as successful;
        # historical failures cannot be reconstructed.
        migration = "backfill_upload_statistics_v1"
        applied = conn.execute(
            "SELECT 1 FROM isolation_migrations WHERE name = ?",
            (migration,),
        ).fetchone()
        if not applied:
            conn.execute(
                """
                INSERT INTO isolation_upload_events (
                  event_id, dataset_id, user_id,
                  submitted_photos, successful_photos, unsuccessful_photos,
                  submitted_pairs, successful_pairs, unsuccessful_pairs,
                  error, created_at
                )
                SELECT
                  'backfill:' || dataset_id, dataset_id, owner_user_id,
                  pair_count * 2, pair_count * 2, 0,
                  pair_count, pair_count, 0,
                  NULL, updated_at
                FROM isolation_datasets
                WHERE pair_count > 0
                """
            )
            conn.execute(
                "INSERT INTO isolation_migrations (name, applied_at) VALUES (?, ?)",
                (migration, float(time.time())),
            )
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


def insert_upload_event(
    db_path: Path,
    *,
    dataset_id: str | None,
    user_id: str | None,
    submitted_photos: int,
    successful_photos: int,
    submitted_pairs: int,
    successful_pairs: int,
    error: str | None = None,
) -> None:
    submitted_photos = max(0, int(submitted_photos))
    successful_photos = min(submitted_photos, max(0, int(successful_photos)))
    submitted_pairs = max(0, int(submitted_pairs))
    successful_pairs = min(submitted_pairs, max(0, int(successful_pairs)))
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO isolation_upload_events (
              event_id, dataset_id, user_id,
              submitted_photos, successful_photos, unsuccessful_photos,
              submitted_pairs, successful_pairs, unsuccessful_pairs,
              error, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                dataset_id,
                user_id,
                submitted_photos,
                successful_photos,
                submitted_photos - successful_photos,
                submitted_pairs,
                successful_pairs,
                submitted_pairs - successful_pairs,
                (error or "")[:1000] or None,
                float(time.time()),
            ),
        )
        conn.commit()


def get_statistics(db_path: Path, *, dataset_id: str | None = None) -> dict[str, Any]:
    event_where = "WHERE e.dataset_id = ?" if dataset_id else ""
    job_where = "WHERE dataset_id = ?" if dataset_id else ""
    dataset_where = "WHERE dataset_id = ?" if dataset_id else ""
    params = (dataset_id,) if dataset_id else ()
    with _connect(db_path) as conn:
        upload = conn.execute(
            f"""
            SELECT
              COUNT(*) AS upload_attempts,
              COALESCE(SUM(submitted_photos), 0) AS submitted_photos,
              COALESCE(SUM(successful_photos), 0) AS successful_photos,
              COALESCE(SUM(unsuccessful_photos), 0) AS unsuccessful_photos,
              COALESCE(SUM(submitted_pairs), 0) AS submitted_pairs,
              COALESCE(SUM(successful_pairs), 0) AS successful_pairs,
              COALESCE(SUM(unsuccessful_pairs), 0) AS unsuccessful_pairs
            FROM isolation_upload_events e
            {event_where}
            """,
            params,
        ).fetchone()
        datasets = conn.execute(
            f"""
            SELECT COUNT(*) AS total_datasets,
                   COALESCE(SUM(pair_count), 0) AS current_pairs
            FROM isolation_datasets
            {dataset_where}
            """,
            params,
        ).fetchone()
        training = conn.execute(
            f"""
            SELECT COUNT(*) AS total,
                   COALESCE(SUM(status = 'completed'), 0) AS completed,
                   COALESCE(SUM(status = 'failed'), 0) AS failed,
                   COALESCE(SUM(status = 'queued'), 0) AS queued,
                   COALESCE(SUM(status = 'running'), 0) AS running
            FROM isolation_train_jobs
            {job_where}
            """,
            params,
        ).fetchone()
        contributors = conn.execute(
            f"""
            SELECT e.user_id, u.email, u.name,
                   COUNT(*) AS upload_attempts,
                   SUM(e.submitted_photos) AS submitted_photos,
                   SUM(e.successful_photos) AS successful_photos,
                   SUM(e.unsuccessful_photos) AS unsuccessful_photos,
                   MAX(e.created_at) AS last_submitted_at
            FROM isolation_upload_events e
            LEFT JOIN users u ON u.user_id = e.user_id
            {event_where}
            GROUP BY e.user_id, u.email, u.name
            ORDER BY last_submitted_at DESC
            """,
            params,
        ).fetchall()

    return {
        "datasets": dict(datasets),
        "uploads": dict(upload),
        "training": dict(training),
        "contributors": [dict(row) for row in contributors],
    }


def insert_train_job(db_path: Path, row: dict[str, Any]) -> dict[str, Any]:
    now = float(time.time())
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO isolation_train_jobs (
              job_id, dataset_id, status, progress, base_model, epochs, val_split,
              model_id, metrics_json, error, provider, owner_user_id,
              resume_from_model_id, grow_active, auto_activate,
              created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                row.get("resume_from_model_id"),
                1 if row.get("grow_active", True) else 0,
                1 if row.get("auto_activate", True) else 0,
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


def upsert_model(db_path: Path, row: dict[str, Any]) -> dict[str, Any]:
    """Insert or update model weights/metrics (used when grow_active reuses model_id)."""
    existing = get_model(db_path, row["model_id"])
    if not existing:
        return insert_model(db_path, row)
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE isolation_models
            SET name = ?, dataset_id = ?, onnx_rel_path = ?, metrics_json = ?,
                is_active = ?, owner_user_id = COALESCE(?, owner_user_id)
            WHERE model_id = ?
            """,
            (
                row.get("name") or existing["name"],
                row.get("dataset_id", existing.get("dataset_id")),
                row["onnx_rel_path"],
                row.get("metrics_json"),
                int(row.get("is_active") if row.get("is_active") is not None else existing.get("is_active") or 0),
                row.get("owner_user_id"),
                row["model_id"],
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
