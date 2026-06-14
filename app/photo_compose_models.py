from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PhotoComposeStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class PhotoComposeResponse(BaseModel):
    compose_id: str
    status: PhotoComposeStatus
    result_url: str | None = None
    result_image_base64: str | None = None
    revised_prompt: str | None = None
    error: str | None = None


def _ts_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def row_to_response(row: dict[str, Any]) -> PhotoComposeResponse:
    return PhotoComposeResponse(
        compose_id=row["compose_id"],
        status=PhotoComposeStatus(row["status"]),
        result_url=row.get("result_url"),
        result_image_base64=None,
        revised_prompt=row.get("revised_prompt"),
        error=row.get("error"),
    )
