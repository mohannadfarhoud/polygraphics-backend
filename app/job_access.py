from __future__ import annotations

import os

from fastapi import Depends, HTTPException

from .auth_deps import get_current_user
from .auth_models import UserPublic


def jobs_require_auth() -> bool:
    return os.getenv("APP_REQUIRE_AUTH_FOR_JOBS", "1").strip().lower() not in ("0", "false", "no")


def require_authenticated_user(
    current_user: UserPublic = Depends(get_current_user),
) -> UserPublic:
    return current_user


def enforce_job_owner(owner_user_id: str | None, user_id: str) -> None:
    if owner_user_id is None:
        if jobs_require_auth():
            raise HTTPException(status_code=403, detail="This job is not assigned to a user")
        return
    if owner_user_id != user_id:
        raise HTTPException(status_code=403, detail="Not authorized for this job")
