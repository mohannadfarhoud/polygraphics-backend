from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from . import quota_db, try_on_db
from . import isolation_db
from .interfaces import JobStatus
from .job_models import JobRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY NOT NULL,
  status TEXT NOT NULL,
  stage TEXT,
  progress INTEGER,
  model_url TEXT,
  model_format TEXT,
  error TEXT,
  image_count INTEGER NOT NULL DEFAULT 0,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
"""


def _ensure_columns(conn: sqlite3.Connection) -> None:
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(jobs)").fetchall()}
    if "model_format" not in cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN model_format TEXT")
    if "progress" not in cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN progress INTEGER")
    if "owner_user_id" not in cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN owner_user_id TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_owner_user_id ON jobs(owner_user_id)")


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(db_path),
        timeout=30.0,
        check_same_thread=False,
    )
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_and_migrate(db_path: Path, root_dir: Path) -> None:
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)
        _ensure_columns(conn)
        conn.commit()
    try_on_db.init_schema(db_path)
    from . import auth_db, photo_compose_db

    auth_db.init_schema(db_path)
    quota_db.init_schema(db_path)
    photo_compose_db.init_schema(db_path)
    isolation_db.init_schema(db_path)

    json_path = root_dir / "config" / "jobs.json"
    if not json_path.is_file():
        return

    with _connect(db_path) as conn:
        n = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        if n > 0:
            return
        raw = json_path.read_text(encoding="utf-8").strip()
        if not raw:
            return
        data = json.loads(raw)
        if not isinstance(data, dict) or not data:
            return
        for _jid, row in data.items():
            try:
                rec = JobRecord.model_validate(row)
                _save_record_conn(conn, rec)
            except Exception:
                continue
        conn.commit()

    bak = json_path.with_suffix(".json.bak")
    if bak.exists():
        try:
            bak.unlink()
        except OSError:
            pass
    try:
        json_path.rename(bak)
    except OSError:
        pass


def _row_to_dict(row: sqlite3.Row) -> dict:
    keys = row.keys()
    progress: int | None = None
    if "progress" in keys and row["progress"] is not None:
        progress = int(row["progress"])
    owner_user_id: str | None = None
    if "owner_user_id" in keys and row["owner_user_id"]:
        owner_user_id = str(row["owner_user_id"])
    return {
        "job_id": row["job_id"],
        "status": row["status"],
        "stage": row["stage"],
        "progress": progress,
        "model_url": row["model_url"],
        "model_format": row["model_format"] if "model_format" in keys else None,
        "error": row["error"],
        "image_count": int(row["image_count"]),
        "created_at": float(row["created_at"]),
        "updated_at": float(row["updated_at"]),
        "owner_user_id": owner_user_id,
    }


def _save_record_conn(conn: sqlite3.Connection, rec: JobRecord) -> None:
    st = rec.status.value if isinstance(rec.status, JobStatus) else str(rec.status)
    conn.execute(
        """
        INSERT OR REPLACE INTO jobs
        (job_id, status, stage, progress, model_url, model_format, error, image_count, created_at, updated_at, owner_user_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            rec.job_id,
            st,
            rec.stage,
            None if rec.progress is None else int(rec.progress),
            rec.model_url,
            rec.model_format,
            rec.error,
            int(rec.image_count),
            float(rec.created_at),
            float(rec.updated_at),
            rec.owner_user_id,
        ),
    )


def get_job(db_path: Path, job_id: str) -> JobRecord | None:
    with _connect(db_path) as conn:
        cur = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        row = cur.fetchone()
        if not row:
            return None
        return JobRecord.model_validate(_row_to_dict(row))


def list_jobs(db_path: Path, *, owner_user_id: str | None = None) -> list[JobRecord]:
    with _connect(db_path) as conn:
        if owner_user_id:
            cur = conn.execute(
                "SELECT * FROM jobs WHERE owner_user_id = ? ORDER BY created_at DESC",
                (owner_user_id,),
            )
        else:
            cur = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC")
        return [JobRecord.model_validate(_row_to_dict(r)) for r in cur.fetchall()]


def get_job_owner(db_path: Path, job_id: str) -> str | None:
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT owner_user_id FROM jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
    if not row or not row["owner_user_id"]:
        return None
    return str(row["owner_user_id"])


def set_job_owner_if_unset(db_path: Path, job_id: str, owner_user_id: str) -> None:
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE jobs SET owner_user_id = ?
            WHERE job_id = ? AND (owner_user_id IS NULL OR owner_user_id = '')
            """,
            (owner_user_id, job_id),
        )
        conn.commit()


def count_jobs_by_status(db_path: Path) -> dict[str, int]:
    """Return ``{status_value: count}`` for all rows in ``jobs``."""
    with _connect(db_path) as conn:
        cur = conn.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status")
        return {str(row[0]): int(row[1]) for row in cur.fetchall()}


def list_queued_oldest_first(db_path: Path) -> list[JobRecord]:
    """Jobs in ``QUEUED`` state, oldest ``created_at`` first (FIFO work queue)."""
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            SELECT * FROM jobs
            WHERE status = ?
            ORDER BY created_at ASC
            """,
            (JobStatus.QUEUED.value,),
        )
        return [JobRecord.model_validate(_row_to_dict(r)) for r in cur.fetchall()]


def save_record(db_path: Path, rec: JobRecord) -> None:
    with _connect(db_path) as conn:
        _save_record_conn(conn, rec)
        conn.commit()


def delete_job(db_path: Path, job_id: str) -> bool:
    """Delete a job row by id. Returns True if a row was removed."""
    with _connect(db_path) as conn:
        cur = conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
        conn.commit()
        return int(cur.rowcount or 0) > 0
