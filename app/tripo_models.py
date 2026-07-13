from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class TripoJobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class TripoImageToModelResponse(BaseModel):
    tripo_job_id: str = Field(description="PolyGraphics Tripo job id (poll this endpoint).")
    status: TripoJobStatus
    download_url: str | None = Field(
        default=None,
        description="Relative API path to download the GLB when status is completed.",
    )
    tripo_task_id: str | None = Field(default=None, description="Tripo cloud task id when known.")
    error: str | None = None


def row_to_response(row: dict, *, download_url: str | None = None) -> TripoImageToModelResponse:
    url = download_url
    if url is None and row.get("status") == TripoJobStatus.completed.value and row.get("model_path"):
        url = None  # caller sets via service
    return TripoImageToModelResponse(
        tripo_job_id=row["tripo_job_id"],
        status=TripoJobStatus(row["status"]),
        download_url=url,
        tripo_task_id=row.get("tripo_task_id"),
        error=row.get("error"),
    )
