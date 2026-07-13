from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import Depends, Header, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .auth_models import UserPublic
from .auth_service import GoogleAuthError, get_user_from_access_token

_db_path: Path | None = None
_bearer = HTTPBearer(auto_error=False)


def init_auth_deps(db_path: Path) -> None:
    global _db_path
    _db_path = db_path


def _require_db_path() -> Path:
    if _db_path is None:
        raise RuntimeError("auth deps not initialized")
    return _db_path


def _extract_bearer_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> str | None:
    if credentials is None or credentials.scheme.lower() != "bearer":
        return None
    token = credentials.credentials.strip()
    return token or None


def get_optional_current_user(
    token: Annotated[str | None, Depends(_extract_bearer_token)],
) -> UserPublic | None:
    if not token:
        return None
    try:
        return get_user_from_access_token(_require_db_path(), token)
    except GoogleAuthError:
        return None


def get_current_user(
    token: Annotated[str | None, Depends(_extract_bearer_token)],
) -> UserPublic:
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        return get_user_from_access_token(_require_db_path(), token)
    except GoogleAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def resolve_user_id(
    current_user: Annotated[UserPublic | None, Depends(get_optional_current_user)],
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
) -> str | None:
    """Prefer verified JWT user id; fall back to legacy X-User-Id header."""
    if current_user is not None:
        return current_user.user_id
    v = (x_user_id or "").strip()
    return v or None


def require_user_id(
    current_user: Annotated[UserPublic, Depends(get_current_user)],
) -> str:
    return current_user.user_id
