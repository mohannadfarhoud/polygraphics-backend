from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class IsolationTrainStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class IsolationDatasetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class IsolationDatasetSummary(BaseModel):
    dataset_id: str
    name: str
    pair_count: int
    created_at: str
    updated_at: str


class IsolationDatasetDetail(IsolationDatasetSummary):
    pairs: list[dict[str, Any]] = Field(default_factory=list)


class IsolationPairUploadResult(BaseModel):
    dataset_id: str
    uploaded: int
    pair_indices: list[int]


class IsolationContributorStatistics(BaseModel):
    user_id: str | None = None
    email: str | None = None
    name: str | None = None
    upload_attempts: int = 0
    submitted_photos: int = 0
    successful_photos: int = 0
    unsuccessful_photos: int = 0
    last_submitted_at: str | None = None


class IsolationUploadStatistics(BaseModel):
    upload_attempts: int = 0
    submitted_photos: int = 0
    successful_photos: int = 0
    unsuccessful_photos: int = 0
    submitted_pairs: int = 0
    successful_pairs: int = 0
    unsuccessful_pairs: int = 0


class IsolationTrainingStatistics(BaseModel):
    total: int = 0
    completed: int = 0
    failed: int = 0
    queued: int = 0
    running: int = 0


class IsolationDatasetStatistics(BaseModel):
    total_datasets: int = 0
    current_pairs: int = 0


class IsolationStatistics(BaseModel):
    dataset_id: str | None = None
    datasets: IsolationDatasetStatistics
    uploads: IsolationUploadStatistics
    training: IsolationTrainingStatistics
    contributors: list[IsolationContributorStatistics] = Field(default_factory=list)


class IsolationTrainRequest(BaseModel):
    dataset_id: str
    base_model: str = Field(default="isnet-general-use", description="rembg-compatible base model name (bootstrap only)")
    epochs: int = Field(default=20, ge=1, le=500)
    val_split: float = Field(default=0.2, ge=0.05, le=0.5)
    force_min_pairs: bool = Field(
        default=False,
        description="Allow training with fewer than 20 pairs (including a single couple).",
    )
    grow_active: bool = Field(
        default=True,
        description="Resume from the active model checkpoint and update that same model_id (one growing model).",
    )
    resume_from_model_id: str | None = Field(
        default=None,
        description="Optional explicit parent model to resume from (overrides grow_active source).",
    )
    auto_activate: bool = Field(
        default=True,
        description="Activate the resulting model when training completes.",
    )


class IsolationTrainMetrics(BaseModel):
    iou: float | None = None
    iou_all: float | None = None
    precision: float | None = None
    recall: float | None = None
    val_pairs: int | None = None
    train_pairs: int | None = None
    base_model: str | None = None
    generation: int | None = None
    backend: str | None = None
    resumed: bool | None = None
    parent_model_id: str | None = None
    val_split_mode: str | None = None
    note: str | None = None

    model_config = {"extra": "ignore"}


class IsolationTrainResponse(BaseModel):
    job_id: str
    status: IsolationTrainStatus
    progress: int = 0
    metrics: IsolationTrainMetrics | None = None
    model_id: str | None = None
    error: str | None = None
    generation: int | None = None
    resumed_from: str | None = None


class IsolationModelSummary(BaseModel):
    model_id: str
    name: str
    dataset_id: str | None = None
    is_active: bool = False
    metrics: IsolationTrainMetrics | None = None
    created_at: str


class IsolationPredictJsonResponse(BaseModel):
    isolated_url: str
    mask_url: str
    model_id: str
    latency_ms: int


class IsolationHealthResponse(BaseModel):
    onnxruntime_ok: bool
    model_loaded: bool
    active_model_id: str | None = None
    active_model_path: str | None = None


def row_to_dataset_summary(row: dict[str, Any]) -> IsolationDatasetSummary:
    from datetime import datetime, timezone

    def _iso(ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return IsolationDatasetSummary(
        dataset_id=row["dataset_id"],
        name=row["name"],
        pair_count=int(row.get("pair_count") or 0),
        created_at=_iso(float(row["created_at"])),
        updated_at=_iso(float(row["updated_at"])),
    )
