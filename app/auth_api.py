from __future__ import annotations

from pathlib import Path
from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse

from .auth_deps import get_current_user
from .auth_models import (
    AuthMessageResponse,
    AuthTokenResponse,
    EmailLoginRequest,
    EmailRegisterRequest,
    GoogleIdTokenRequest,
    UserPublic,
)
from .auth_service import (
    AuthConfigError,
    EmailAuthError,
    GoogleAuthError,
    auth_frontend_callback_url,
    build_google_login_url,
    login_or_register_with_google_code,
    login_or_register_with_google_id_token,
    login_with_email,
    new_oauth_state,
    register_with_email,
)

router = APIRouter(prefix="/auth", tags=["auth"])

_db_path: Path | None = None
_oauth_states: dict[str, float] = {}
_STATE_TTL_SECONDS = 600.0


def init_auth_api(db_path: Path) -> None:
    global _db_path
    _db_path = db_path


def _require_db_path() -> Path:
    if _db_path is None:
        raise RuntimeError("auth_api not initialized")
    return _db_path


def _remember_state(state: str) -> None:
    import time

    now = time.time()
    _oauth_states[state] = now
    expired = [key for key, ts in _oauth_states.items() if now - ts > _STATE_TTL_SECONDS]
    for key in expired:
        _oauth_states.pop(key, None)


def _consume_state(state: str | None) -> None:
    if not state or state not in _oauth_states:
        raise HTTPException(status_code=400, detail="Invalid OAuth state")
    _oauth_states.pop(state, None)


@router.get("/google/login")
def google_login_redirect() -> RedirectResponse:
    """Start Google OAuth (browser redirect)."""
    try:
        state = new_oauth_state()
        _remember_state(state)
        return RedirectResponse(build_google_login_url(state=state), status_code=302)
    except AuthConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/google/callback")
def google_oauth_callback(
    code: Annotated[str | None, Query()] = None,
    state: Annotated[str | None, Query()] = None,
    redirect: Annotated[bool, Query(description="When true, redirect to frontend with token")] = True,
):
    """Complete Google OAuth redirect flow and issue an API access token."""
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code")
    _consume_state(state)
    try:
        auth = login_or_register_with_google_code(_require_db_path(), code)
    except AuthConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GoogleAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    if not redirect:
        return auth

    callback = auth_frontend_callback_url()
    query = urlencode(
        {
            "access_token": auth.access_token,
            "token_type": auth.token_type,
            "expires_in": str(auth.expires_in),
        }
    )
    sep = "&" if "?" in callback else "?"
    return RedirectResponse(f"{callback}{sep}{query}", status_code=302)


@router.post("/google", response_model=AuthTokenResponse)
def google_sign_in(body: GoogleIdTokenRequest) -> AuthTokenResponse:
    """Register or log in with a Google Sign-In ID token from the frontend."""
    token = (body.id_token or body.credential or "").strip()
    if not token:
        raise HTTPException(status_code=400, detail="id_token or credential is required")
    try:
        return login_or_register_with_google_id_token(_require_db_path(), token)
    except AuthConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GoogleAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@router.post("/register", response_model=AuthTokenResponse)
def email_register(body: EmailRegisterRequest) -> AuthTokenResponse:
    """Create an account with email and password (no email verification)."""
    try:
        return register_with_email(
            _require_db_path(),
            email=body.email,
            password=body.password,
            name=body.name,
        )
    except AuthConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except EmailAuthError as exc:
        msg = str(exc)
        if "already registered" in msg.lower():
            raise HTTPException(status_code=409, detail=msg) from exc
        raise HTTPException(status_code=400, detail=msg) from exc


@router.post("/login", response_model=AuthTokenResponse)
def email_login(body: EmailLoginRequest) -> AuthTokenResponse:
    """Sign in with email and password."""
    try:
        return login_with_email(_require_db_path(), email=body.email, password=body.password)
    except AuthConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except EmailAuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@router.get("/me", response_model=UserPublic)
def auth_me(current_user: UserPublic = Depends(get_current_user)) -> UserPublic:
    return current_user


@router.post("/logout", response_model=AuthMessageResponse)
def auth_logout(_current_user: UserPublic = Depends(get_current_user)) -> AuthMessageResponse:
    return AuthMessageResponse(message="Signed out. Discard the access token on the client.")
