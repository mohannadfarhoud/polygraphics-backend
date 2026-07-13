from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from . import quota_db


class UserQuotaStatus(BaseModel):
    used: int = Field(description="Reconstructions started this calendar month (UTC).")
    limit: int = Field(description="Free monthly reconstruction limit.")
    remaining: int
    period_start: str = Field(description="UTC month start (ISO date).")
    period_end: str = Field(description="UTC month end exclusive (ISO date).")


def free_models_per_month() -> int:
    raw = os.getenv("APP_FREE_MODELS_PER_MONTH", "10").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 10
    return max(1, value)


def get_user_quota(db_path: Path, user_id: str) -> UserQuotaStatus:
    used = quota_db.count_user_starts_in_current_month(db_path, user_id)
    limit = free_models_per_month()
    year, month, start_ts, end_ts = quota_db.current_month_bounds_utc()
    period_start = datetime.fromtimestamp(start_ts, tz=timezone.utc).strftime("%Y-%m-%d")
    period_end = datetime.fromtimestamp(end_ts, tz=timezone.utc).strftime("%Y-%m-%d")
    return UserQuotaStatus(
        used=used,
        limit=limit,
        remaining=max(0, limit - used),
        period_start=period_start,
        period_end=period_end,
    )


class QuotaExceededError(RuntimeError):
    pass


def assert_can_start_reconstruction(db_path: Path, *, user_id: str, job_id: str) -> None:
    if quota_db.job_already_charged_this_month(db_path, user_id=user_id, job_id=job_id):
        return
    quota = get_user_quota(db_path, user_id)
    if quota.used >= quota.limit:
        raise QuotaExceededError(
            f"Monthly free limit reached ({quota.limit} reconstructions). "
            f"Resets on {quota.period_end}."
        )


def charge_reconstruction_start(db_path: Path, *, user_id: str, job_id: str) -> None:
    if quota_db.job_already_charged_this_month(db_path, user_id=user_id, job_id=job_id):
        return
    assert_can_start_reconstruction(db_path, user_id=user_id, job_id=job_id)
    quota_db.record_reconstruction_start(db_path, user_id=user_id, job_id=job_id)
