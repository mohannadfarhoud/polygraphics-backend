from __future__ import annotations

import os
import secrets
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
import jwt

from .auth_db import (
    create_email_user,
    get_user_by_email,
    get_user_by_id,
    is_valid_email,
    normalize_email,
    touch_user_login,
    upsert_google_user,
)
from .auth_models import AuthTokenResponse, UserPublic

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"


class AuthConfigError(RuntimeError):
    pass


class GoogleAuthError(ValueError):
    pass


class EmailAuthError(ValueError):
    pass


def _jwt_secret() -> str:
    secret = os.getenv("APP_JWT_SECRET", "").strip()
    if not secret:
        raise AuthConfigError("APP_JWT_SECRET is not set")
    return secret


def jwt_expiry_seconds() -> int:
    raw = os.getenv("APP_JWT_EXPIRY_HOURS", "720").strip()
    try:
        hours = float(raw)
    except ValueError:
        hours = 720.0
    return max(60, int(hours * 3600))


def google_client_id() -> str:
    client_id = os.getenv("APP_GOOGLE_CLIENT_ID", "").strip()
    if not client_id:
        raise AuthConfigError("APP_GOOGLE_CLIENT_ID is not set")
    return client_id


def google_client_secret() -> str:
    secret = os.getenv("APP_GOOGLE_CLIENT_SECRET", "").strip()
    if not secret:
        raise AuthConfigError("APP_GOOGLE_CLIENT_SECRET is not set")
    return secret


def google_redirect_uri() -> str:
    explicit = os.getenv("APP_GOOGLE_REDIRECT_URI", "").strip()
    if explicit:
        return explicit.rstrip("/")
    public_base = os.getenv("APP_PUBLIC_BASE_URL", "").strip().rstrip("/")
    if public_base:
        return f"{public_base}/auth/google/callback"
    root_path = os.getenv("APP_ROOT_PATH", "").strip().rstrip("/")
    if root_path and not root_path.startswith("/"):
        root_path = "/" + root_path
    return f"{root_path}/auth/google/callback".replace("//", "/")


def auth_frontend_callback_url() -> str:
    url = os.getenv("APP_AUTH_FRONTEND_CALLBACK", "").strip()
    if url:
        return url.rstrip("/")
    frontend = os.getenv("APP_AUTH_FRONTEND_URL", "").strip().rstrip("/")
    if frontend:
        return f"{frontend}/auth/callback"
    return "/auth/callback"


def user_to_public(row: dict[str, Any]) -> UserPublic:
    return UserPublic(
        user_id=row["user_id"],
        email=row["email"],
        name=row.get("name"),
        picture_url=row.get("picture_url"),
    )


def issue_access_token(user: dict[str, Any]) -> AuthTokenResponse:
    now = int(time.time())
    expires_in = jwt_expiry_seconds()
    payload = {
        "sub": user["user_id"],
        "email": user["email"],
        "iat": now,
        "exp": now + expires_in,
    }
    token = jwt.encode(payload, _jwt_secret(), algorithm="HS256")
    return AuthTokenResponse(
        access_token=token,
        expires_in=expires_in,
        user=user_to_public(user),
    )


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(token, _jwt_secret(), algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise GoogleAuthError("Invalid or expired access token") from exc
    user_id = str(payload.get("sub") or "").strip()
    if not user_id:
        raise GoogleAuthError("Invalid access token payload")
    return payload


def get_user_from_access_token(db_path: Path, token: str) -> UserPublic:
    payload = decode_access_token(token)
    row = get_user_by_id(db_path, str(payload["sub"]))
    if row is None:
        raise GoogleAuthError("User not found")
    return user_to_public(row)


def verify_google_id_token(id_token_value: str) -> dict[str, Any]:
    try:
        from google.auth.transport import requests as google_requests
        from google.oauth2 import id_token as google_id_token
    except ImportError as exc:
        raise AuthConfigError("google-auth is not installed") from exc

    try:
        info = google_id_token.verify_oauth2_token(
            id_token_value,
            google_requests.Request(),
            google_client_id(),
        )
    except Exception as exc:
        raise GoogleAuthError("Invalid Google ID token") from exc

    sub = str(info.get("sub") or "").strip()
    email = str(info.get("email") or "").strip()
    if not sub or not email:
        raise GoogleAuthError("Google token missing required profile fields")
    if info.get("email_verified") is False:
        raise GoogleAuthError("Google email is not verified")
    return info


def login_or_register_with_google_id_token(db_path: Path, id_token_value: str) -> AuthTokenResponse:
    info = verify_google_id_token(id_token_value)
    user = upsert_google_user(
        db_path,
        google_sub=str(info["sub"]),
        email=str(info["email"]),
        name=(str(info["name"]).strip() if info.get("name") else None),
        picture_url=(str(info["picture"]).strip() if info.get("picture") else None),
    )
    return issue_access_token(user)


def build_google_login_url(*, state: str) -> str:
    params = {
        "client_id": google_client_id(),
        "redirect_uri": google_redirect_uri(),
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "online",
        "include_granted_scopes": "true",
        "prompt": "select_account",
        "state": state,
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"


def new_oauth_state() -> str:
    return secrets.token_urlsafe(24)


def exchange_google_code_for_tokens(code: str) -> dict[str, Any]:
    payload = {
        "code": code,
        "client_id": google_client_id(),
        "client_secret": google_client_secret(),
        "redirect_uri": google_redirect_uri(),
        "grant_type": "authorization_code",
    }
    with httpx.Client(timeout=30.0) as client:
        response = client.post(GOOGLE_TOKEN_URL, data=payload)
    if response.status_code >= 400:
        raise GoogleAuthError(f"Google token exchange failed ({response.status_code})")
    data = response.json()
    id_token_value = str(data.get("id_token") or "").strip()
    if not id_token_value:
        raise GoogleAuthError("Google token response missing id_token")
    return data


def login_or_register_with_google_code(db_path: Path, code: str) -> AuthTokenResponse:
    data = exchange_google_code_for_tokens(code)
    return login_or_register_with_google_id_token(db_path, str(data["id_token"]))


def _hash_password(password: str) -> str:
    import bcrypt

    hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    import bcrypt

    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def register_with_email(
    db_path: Path,
    *,
    email: str,
    password: str,
    name: str | None = None,
) -> AuthTokenResponse:
    normalized = normalize_email(email)
    if not is_valid_email(normalized):
        raise EmailAuthError("Invalid email address")
    if len(password) < 8:
        raise EmailAuthError("Password must be at least 8 characters")
    try:
        user = create_email_user(
            db_path,
            email=normalized,
            password_hash=_hash_password(password),
            name=(name.strip() if name else None),
        )
    except ValueError as exc:
        raise EmailAuthError(str(exc)) from exc
    return issue_access_token(user)


def login_with_email(db_path: Path, *, email: str, password: str) -> AuthTokenResponse:
    normalized = normalize_email(email)
    user = get_user_by_email(db_path, normalized)
    if user is None:
        raise EmailAuthError("Invalid email or password")
    password_hash = user.get("password_hash")
    if not password_hash or not _verify_password(password, str(password_hash)):
        raise EmailAuthError("Invalid email or password")
    touch_user_login(db_path, user["user_id"])
    refreshed = get_user_by_id(db_path, user["user_id"])
    if refreshed is None:
        raise EmailAuthError("User not found")
    return issue_access_token(refreshed)
