from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except Exception:
        return None


class CaptureFrameMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    file: str | None = None
    frame_index: int | None = Field(default=None, ge=0)
    timestamp_ms: int | None = Field(default=None, ge=0)
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None
    depth_m: float | None = None
    distance_m: float | None = None
    blur_score: float | None = None
    exposure_score: float | None = None
    quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    accepted: bool | None = None


class CaptureMetadataSummary(BaseModel):
    model_config = ConfigDict(extra="allow")

    total_frames: int | None = Field(default=None, ge=0)
    accepted_frames: int | None = Field(default=None, ge=0)
    rejected_frames: int | None = Field(default=None, ge=0)
    avg_quality_score: float | None = Field(default=None, ge=0.0, le=1.0)
    orbit_coverage_deg: float | None = Field(default=None, ge=0.0, le=360.0)


class CaptureMetadataPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    schema_version: str = "1.0"
    session_id: str | None = None
    session: dict[str, Any] = Field(default_factory=dict)
    device: dict[str, Any] = Field(default_factory=dict)
    frames: list[CaptureFrameMetadata] = Field(default_factory=list)
    summary: CaptureMetadataSummary | None = None


def _derive_summary(frames: list[CaptureFrameMetadata]) -> CaptureMetadataSummary:
    total = len(frames)
    accepted = sum(1 for f in frames if f.accepted is not False)
    rejected = max(0, total - accepted)

    quality_values = [f.quality_score for f in frames if f.quality_score is not None]
    avg_quality = float(mean(quality_values)) if quality_values else None

    yaw_values = [f.yaw_deg for f in frames if f.yaw_deg is not None]
    coverage: float | None = None
    if len(yaw_values) >= 2:
        ymin = min(float(y) for y in yaw_values)
        ymax = max(float(y) for y in yaw_values)
        coverage = max(0.0, min(360.0, ymax - ymin))

    return CaptureMetadataSummary(
        total_frames=total,
        accepted_frames=accepted,
        rejected_frames=rejected,
        avg_quality_score=avg_quality,
        orbit_coverage_deg=coverage,
    )


def validate_capture_metadata_payload(raw_payload: Any) -> dict[str, Any]:
    """Validate and normalize app-provided capture metadata payload."""
    payload = CaptureMetadataPayload.model_validate(raw_payload)
    normalized = payload.model_dump(mode="json", exclude_none=True)

    # Ensure we always persist a summary block, even when app omitted it.
    summary = payload.summary or _derive_summary(payload.frames)
    normalized["summary"] = summary.model_dump(mode="json", exclude_none=True)
    return normalized


def parse_capture_metadata_form(raw: str | None) -> dict[str, Any] | None:
    if raw is None or not raw.strip():
        return None
    try:
        decoded = json.loads(raw)
    except Exception as exc:
        raise ValueError("capture_metadata must be valid JSON text in multipart form data.") from exc
    if not isinstance(decoded, dict):
        raise ValueError("capture_metadata JSON must be an object.")
    try:
        return validate_capture_metadata_payload(decoded)
    except Exception as exc:
        raise ValueError(f"capture_metadata validation failed: {exc}") from exc


def capture_metadata_path(upload_dir: Path, job_id: str) -> Path:
    return upload_dir / job_id / "capture_metadata.json"


def write_capture_metadata(upload_dir: Path, job_id: str, payload: dict[str, Any]) -> Path:
    out_path = capture_metadata_path(upload_dir, job_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def read_capture_metadata(upload_dir: Path, job_id: str) -> dict[str, Any] | None:
    p = capture_metadata_path(upload_dir, job_id)
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def capture_metadata_response_fields(
    payload: dict[str, Any] | None,
) -> tuple[str | None, int | None, int | None, float | None, float | None]:
    if not payload:
        return None, None, None, None, None
    version = payload.get("schema_version")
    if not isinstance(version, str):
        version = None

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        summary = {}

    total_raw = summary.get("total_frames")
    accepted_raw = summary.get("accepted_frames")
    avg_quality_raw = summary.get("avg_quality_score")
    coverage_raw = summary.get("orbit_coverage_deg")

    total = int(total_raw) if isinstance(total_raw, int) else None
    accepted = int(accepted_raw) if isinstance(accepted_raw, int) else None
    avg_quality = _float_or_none(avg_quality_raw)
    coverage = _float_or_none(coverage_raw)
    return version, total, accepted, avg_quality, coverage
