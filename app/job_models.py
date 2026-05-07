from __future__ import annotations

import time
from datetime import datetime, timezone

from pydantic import BaseModel, Field, computed_field

from .interfaces import JobStatus


class JobRecord(BaseModel):
    job_id: str
    status: JobStatus
    stage: str | None = None
    model_url: str | None = None
    model_format: str | None = None  # "glb" | "ply" | None
    error: str | None = None
    image_count: int = 0
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    @computed_field  # type: ignore[misc]
    @property
    def created_at_iso(self) -> str:
        return datetime.fromtimestamp(self.created_at, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    @computed_field  # type: ignore[misc]
    @property
    def updated_at_iso(self) -> str:
        return datetime.fromtimestamp(self.updated_at, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ModelListItem(BaseModel):
    job_id: str
    filename: str
    path: str
    size_bytes: int
    modified_at: float

    @computed_field  # type: ignore[misc]
    @property
    def modified_at_iso(self) -> str:
        return datetime.fromtimestamp(self.modified_at, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
