"""Fixed admin account bootstrap for isolation (legacy trainer username)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import Depends, HTTPException

from .auth_db import (
    create_email_user,
    get_user_by_email,
    update_user_email,
    update_user_password_hash,
)
from .auth_deps import get_current_user
from .auth_models import UserPublic
from .auth_service import _hash_password, _verify_password

log = logging.getLogger(__name__)

LEGACY_TRAINER_EMAIL = "trainer@polygraph.local"

# Canonical shared admin account used by PicPolish training UI.
TRAINER_EMAIL = os.getenv("ISOLATION_TRAINER_EMAIL", "admin@polygraph.local").strip().lower()
TRAINER_PASSWORD = os.getenv("ISOLATION_TRAINER_PASSWORD", "devtek2026").strip()
TRAINER_NAME = os.getenv("ISOLATION_TRAINER_NAME", "admin").strip() or "admin"

# Login aliases accepted by POST /auth/login (in addition to TRAINER_EMAIL).
TRAINER_LOGIN_ALIASES = {
    "admin",
    TRAINER_EMAIL,
    TRAINER_NAME.lower(),
    # Legacy username from earlier deployments.
    "trainer",
    LEGACY_TRAINER_EMAIL,
}


def resolve_login_email(email_or_username: str) -> str:
    """Map username 'admin' (or legacy 'trainer') to the fixed admin email."""
    raw = (email_or_username or "").strip().lower()
    if raw in TRAINER_LOGIN_ALIASES:
        return TRAINER_EMAIL
    return raw


def _migrate_legacy_trainer_user(db_path: Path) -> dict | None:
    """Rename trainer@polygraph.local → admin@polygraph.local when upgrading."""
    if TRAINER_EMAIL == LEGACY_TRAINER_EMAIL:
        return get_user_by_email(db_path, TRAINER_EMAIL)
    legacy = get_user_by_email(db_path, LEGACY_TRAINER_EMAIL)
    if legacy is None:
        return None
    if get_user_by_email(db_path, TRAINER_EMAIL) is not None:
        return get_user_by_email(db_path, TRAINER_EMAIL)
    update_user_email(
        db_path,
        user_id=str(legacy["user_id"]),
        email=TRAINER_EMAIL,
        name=TRAINER_NAME,
    )
    log.info("migrated isolation user %s -> %s", LEGACY_TRAINER_EMAIL, TRAINER_EMAIL)
    return get_user_by_email(db_path, TRAINER_EMAIL)


def ensure_trainer_user(db_path: Path) -> dict:
    """Create or reset the fixed admin user so password always matches env/default."""
    if len(TRAINER_PASSWORD) < 8:
        raise RuntimeError("ISOLATION_TRAINER_PASSWORD must be at least 8 characters")
    existing = _migrate_legacy_trainer_user(db_path) or get_user_by_email(db_path, TRAINER_EMAIL)
    password_hash = _hash_password(TRAINER_PASSWORD)
    if existing is None:
        user = create_email_user(
            db_path,
            email=TRAINER_EMAIL,
            password_hash=password_hash,
            name=TRAINER_NAME,
        )
        log.info("created isolation admin user %s", TRAINER_EMAIL)
        return user

    current_hash = str(existing.get("password_hash") or "")
    current_name = str(existing.get("name") or "")
    needs_password = not current_hash or not _verify_password(TRAINER_PASSWORD, current_hash)
    needs_name = current_name != TRAINER_NAME
    if needs_password or needs_name:
        update_user_password_hash(
            db_path,
            email=TRAINER_EMAIL,
            password_hash=password_hash,
            name=TRAINER_NAME,
        )
        log.info("updated isolation admin account for %s", TRAINER_EMAIL)
        refreshed = get_user_by_email(db_path, TRAINER_EMAIL)
        assert refreshed is not None
        return refreshed
    return existing


def is_admin_email(email: str | None) -> bool:
    """True for the fixed admin account (admin / admin@polygraph.local)."""
    return (email or "").strip().lower() == TRAINER_EMAIL


def require_admin(user: UserPublic = Depends(get_current_user)) -> UserPublic:
    """Only the fixed admin account may access the admin panel APIs."""
    if not is_admin_email(user.email):
        raise HTTPException(
            status_code=403,
            detail="Admin account required. Login as user 'admin'.",
        )
    return user


# Backwards-compatible alias used by older deploy scripts / imports.
require_trainer = require_admin
