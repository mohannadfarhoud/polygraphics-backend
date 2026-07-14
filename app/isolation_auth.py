"""Fixed trainer account for isolation upload/train endpoints."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import Depends, HTTPException

from .auth_db import create_email_user, get_user_by_email, update_user_password_hash
from .auth_deps import get_current_user
from .auth_models import UserPublic
from .auth_service import _hash_password, _verify_password

log = logging.getLogger(__name__)

# Canonical account used by PicPolish training UI.
TRAINER_EMAIL = os.getenv("ISOLATION_TRAINER_EMAIL", "trainer@polygraph.local").strip().lower()
TRAINER_PASSWORD = os.getenv("ISOLATION_TRAINER_PASSWORD", "devtek2026").strip()
TRAINER_NAME = os.getenv("ISOLATION_TRAINER_NAME", "trainer").strip() or "trainer"

# Login aliases accepted by POST /auth/login (in addition to TRAINER_EMAIL).
TRAINER_LOGIN_ALIASES = {
    "trainer",
    TRAINER_EMAIL,
    TRAINER_NAME.lower(),
}


def resolve_login_email(email_or_username: str) -> str:
    """Map username 'trainer' to the fixed trainer email."""
    raw = (email_or_username or "").strip().lower()
    if raw in TRAINER_LOGIN_ALIASES or raw == "trainer":
        return TRAINER_EMAIL
    return raw


def ensure_trainer_user(db_path: Path) -> dict:
    """Create or reset the fixed trainer user so password always matches env/default."""
    if len(TRAINER_PASSWORD) < 8:
        raise RuntimeError("ISOLATION_TRAINER_PASSWORD must be at least 8 characters")
    existing = get_user_by_email(db_path, TRAINER_EMAIL)
    password_hash = _hash_password(TRAINER_PASSWORD)
    if existing is None:
        user = create_email_user(
            db_path,
            email=TRAINER_EMAIL,
            password_hash=password_hash,
            name=TRAINER_NAME,
        )
        log.info("created isolation trainer user %s", TRAINER_EMAIL)
        return user

    # Keep password in sync with configured value (fixed shared trainer account).
    current_hash = str(existing.get("password_hash") or "")
    if not current_hash or not _verify_password(TRAINER_PASSWORD, current_hash):
        update_user_password_hash(
            db_path,
            email=TRAINER_EMAIL,
            password_hash=password_hash,
            name=TRAINER_NAME,
        )
        log.info("updated isolation trainer password for %s", TRAINER_EMAIL)
        refreshed = get_user_by_email(db_path, TRAINER_EMAIL)
        assert refreshed is not None
        return refreshed
    return existing


def require_trainer(user: UserPublic = Depends(get_current_user)) -> UserPublic:
    """Only the fixed trainer account may manage datasets / training uploads."""
    email = (user.email or "").strip().lower()
    if email != TRAINER_EMAIL:
        raise HTTPException(
            status_code=403,
            detail="Trainer account required. Login as user 'trainer'.",
        )
    return user
