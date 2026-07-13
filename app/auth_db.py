from __future__ import annotations

import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
  user_id TEXT PRIMARY KEY NOT NULL,
  google_sub TEXT NOT NULL UNIQUE,
  email TEXT NOT NULL,
  name TEXT,
  picture_url TEXT,
  password_hash TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  last_login_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
"""

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def normalize_email(email: str) -> str:
    return email.strip().lower()


def is_valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match(normalize_email(email)))


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_columns(conn: sqlite3.Connection) -> None:
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "password_hash" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_unique ON users(email)")


def init_schema(db_path: Path) -> None:
    with _connect(db_path) as conn:
        conn.executescript(SCHEMA)
        _ensure_columns(conn)
        conn.commit()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    keys = row.keys()
    return {
        "user_id": row["user_id"],
        "google_sub": row["google_sub"],
        "email": row["email"],
        "name": row["name"],
        "picture_url": row["picture_url"],
        "password_hash": row["password_hash"] if "password_hash" in keys else None,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "last_login_at": row["last_login_at"],
    }


def get_user_by_id(db_path: Path, user_id: str) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
    return _row_to_dict(row) if row else None


def get_user_by_google_sub(db_path: Path, google_sub: str) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE google_sub = ?", (google_sub,)).fetchone()
    return _row_to_dict(row) if row else None


def get_user_by_email(db_path: Path, email: str) -> dict[str, Any] | None:
    normalized = normalize_email(email)
    with _connect(db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (normalized,)).fetchone()
    return _row_to_dict(row) if row else None


def create_email_user(
    db_path: Path,
    *,
    email: str,
    password_hash: str,
    name: str | None = None,
) -> dict[str, Any]:
    normalized = normalize_email(email)
    if get_user_by_email(db_path, normalized):
        raise ValueError("Email is already registered")
    now = time.time()
    user_id = str(uuid.uuid4())
    google_sub = f"local:{user_id}"
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO users (
              user_id, google_sub, email, name, picture_url, password_hash,
              created_at, updated_at, last_login_at
            ) VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?)
            """,
            (user_id, google_sub, normalized, name, password_hash, now, now, now),
        )
        conn.commit()
    user = get_user_by_id(db_path, user_id)
    if user is None:
        raise RuntimeError("Failed to persist email user")
    return user


def touch_user_login(db_path: Path, user_id: str) -> None:
    now = time.time()
    with _connect(db_path) as conn:
        conn.execute(
            "UPDATE users SET last_login_at = ?, updated_at = ? WHERE user_id = ?",
            (now, now, user_id),
        )
        conn.commit()


def upsert_google_user(
    db_path: Path,
    *,
    google_sub: str,
    email: str,
    name: str | None,
    picture_url: str | None,
) -> dict[str, Any]:
    now = time.time()
    normalized = normalize_email(email)
    existing = get_user_by_google_sub(db_path, google_sub)
    if not existing:
        by_email = get_user_by_email(db_path, normalized)
        if by_email:
            existing = by_email
    with _connect(db_path) as conn:
        if existing:
            conn.execute(
                """
                UPDATE users
                SET google_sub = ?, email = ?, name = ?, picture_url = ?, updated_at = ?, last_login_at = ?
                WHERE user_id = ?
                """,
                (google_sub, normalized, name, picture_url, now, now, existing["user_id"]),
            )
            user_id = existing["user_id"]
        else:
            user_id = str(uuid.uuid4())
            conn.execute(
                """
                INSERT INTO users (
                  user_id, google_sub, email, name, picture_url, password_hash,
                  created_at, updated_at, last_login_at
                ) VALUES (?, ?, ?, ?, ?, NULL, ?, ?, ?)
                """,
                (user_id, google_sub, normalized, name, picture_url, now, now, now),
            )
        conn.commit()
    user = get_user_by_id(db_path, user_id)
    if user is None:
        raise RuntimeError("Failed to persist Google user")
    return user
