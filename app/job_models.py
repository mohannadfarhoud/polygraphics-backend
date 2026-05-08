from __future__ import annotations

import time
from datetime import datetime, timezone

from pydantic import BaseModel, Field, computed_field

from .interfaces import JobStatus


class JobRecord(BaseModel):
    job_id: str
    status: JobStatus
    stage: str | None = Field(
        default=None,
        description=(
            "Current pipeline stage. UI should switch on the base value (before any space). "
            "Lifecycle: starting | exporting | completed. "
            "Pipeline protocol: phase_1_segmentation | phase_2_alignment | phase_3_sanitization "
            "| phase_4_colmap_bridge | phase_4_colmap_scene | phase_5_gaussian_splatting. "
            "Mesh-only: meshing. May include a human suffix in parens, e.g. 'phase_1_segmentation (3/27)'."
        ),
    )
    progress: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Pipeline progress in percent (0-100). Null until processing starts.",
    )
    model_url: str | None = None
    model_format: str | None = None  # "glb" | "ply" | None
    error: str | None = None
    image_count: int = 0
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    image_sample_url: str | None = Field(
        default=None,
        description="URL of the first uploaded image for this job (preview thumbnail). Relative by default; absolute when APP_MODEL_BASE_URL is set to a non-loopback URL. Derived per response; not persisted.",
    )

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
    url: str = Field(
        description="Download URL for this asset (same rules as job.model_url: relative /output/... or APP_MODEL_BASE_URL).",
    )
    image_url: str | None = Field(
        default=None,
        description="URL of the first uploaded input image for this job (preview). Same construction as job.image_sample_url.",
    )
    size_bytes: int
    modified_at: float

    @computed_field  # type: ignore[misc]
    @property
    def modified_at_iso(self) -> str:
        return datetime.fromtimestamp(self.modified_at, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
