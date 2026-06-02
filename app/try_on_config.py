"""Virtual try-on configuration: models, validation, and persistence helpers."""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from . import try_on_db

MIN_PROFILE_DISTANCE = 0.02
POINT_CLAMP = 1.0


class JewelryType(str, Enum):
    stud = "stud"
    drop = "drop"


class ModelPoint3D(BaseModel):
    """Point in normalized try-on model space (matches frontend ``prepareTryOnModel``).

    The mesh is centered at the origin and the largest axis of its bounding box is 1.0 unit.
    Components are clamped to [-1, 1] on write.
    """

    x: float = Field(description="X in normalized try-on space, clamped to [-1, 1].")
    y: float = Field(description="Y in normalized try-on space, clamped to [-1, 1].")
    z: float = Field(description="Z in normalized try-on space, clamped to [-1, 1].")

    @field_validator("x", "y", "z", mode="before")
    @classmethod
    def _coerce_float(cls, v: Any) -> float:
        return float(v)


class TryOnJewelryRotation(BaseModel):
    spin_deg: float = Field(default=0.0, description="Yaw around hanger, degrees [-180, 180].")
    pitch_deg: float = Field(default=0.0, description="Pitch, degrees [-75, 75].")
    roll_deg: float = Field(default=0.0, description="Roll, degrees [-180, 180].")


class TryOnEarOffsets(BaseModel):
    vertical: float = Field(default=0.0, description="Fine-tune vertical placement on face.")
    depth: float = Field(default=0.0, description="Fine-tune depth placement on face.")
    lateral: float = Field(default=0.0, description="Fine-tune lateral placement on face.")


class TryOnSideLateralNudge(BaseModel):
    left: float = Field(default=0.0, description="Per-ear lateral nudge (left ear).")
    right: float = Field(default=0.0, description="Per-ear lateral nudge (right ear).")


class ModelTryOnConfig(BaseModel):
    job_id: str
    hanger_point: ModelPoint3D
    profile_point: ModelPoint3D | None = None
    jewelry_rotation: TryOnJewelryRotation | None = None
    api_vertical_flip: bool = False
    ear_offsets: TryOnEarOffsets | None = None
    side_lateral_nudge: TryOnSideLateralNudge | None = None
    jewelry_type: JewelryType = JewelryType.drop
    version: int = Field(default=1, ge=1)
    updated_at: str = Field(
        description="UTC timestamp ISO-8601 (e.g. 2026-06-02T12:00:00Z).",
    )


class ModelTryOnConfigPut(BaseModel):
    """Upsert body. Omitted fields are merged with the existing row on PUT."""

    hanger_point: ModelPoint3D | None = None
    profile_point: ModelPoint3D | None = None
    jewelry_rotation: TryOnJewelryRotation | None = None
    api_vertical_flip: bool | None = None
    ear_offsets: TryOnEarOffsets | None = None
    side_lateral_nudge: TryOnSideLateralNudge | None = None
    jewelry_type: JewelryType | None = None


class HangerPointBody(BaseModel):
    """Backward-compatible hanger-only body (``{ x, y, z }``)."""

    x: float
    y: float
    z: float


class UserTryOnCalibration(BaseModel):
    left_offset: TryOnEarOffsets
    right_offset: TryOnEarOffsets
    updated_at: str


class UserTryOnCalibrationPut(BaseModel):
    left_offset: TryOnEarOffsets | None = None
    right_offset: TryOnEarOffsets | None = None


def clamp_point(p: ModelPoint3D) -> ModelPoint3D:
    return ModelPoint3D(
        x=max(-POINT_CLAMP, min(POINT_CLAMP, float(p.x))),
        y=max(-POINT_CLAMP, min(POINT_CLAMP, float(p.y))),
        z=max(-POINT_CLAMP, min(POINT_CLAMP, float(p.z))),
    )


def clamp_rotation(r: TryOnJewelryRotation) -> TryOnJewelryRotation:
    return TryOnJewelryRotation(
        spin_deg=max(-180.0, min(180.0, float(r.spin_deg))),
        pitch_deg=max(-75.0, min(75.0, float(r.pitch_deg))),
        roll_deg=max(-180.0, min(180.0, float(r.roll_deg))),
    )


def point_distance(a: ModelPoint3D, b: ModelPoint3D) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def validate_profile_distance(hanger: ModelPoint3D, profile: ModelPoint3D | None) -> None:
    if profile is None:
        return
    d = point_distance(hanger, profile)
    if d < MIN_PROFILE_DISTANCE:
        raise ValueError(
            f"profile_point must be at least {MIN_PROFILE_DISTANCE} units from hanger_point "
            f"(distance={d:.4f})."
        )


def _ts_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def row_to_config(row: dict[str, Any]) -> ModelTryOnConfig:
    profile = None
    if (
        row.get("profile_x") is not None
        and row.get("profile_y") is not None
        and row.get("profile_z") is not None
    ):
        profile = ModelPoint3D(x=row["profile_x"], y=row["profile_y"], z=row["profile_z"])
    return ModelTryOnConfig(
        job_id=row["job_id"],
        hanger_point=ModelPoint3D(x=row["hanger_x"], y=row["hanger_y"], z=row["hanger_z"]),
        profile_point=profile,
        jewelry_rotation=TryOnJewelryRotation(
            spin_deg=row.get("spin_deg") or 0,
            pitch_deg=row.get("pitch_deg") or 0,
            roll_deg=row.get("roll_deg") or 0,
        ),
        api_vertical_flip=bool(row.get("api_vertical_flip")),
        ear_offsets=TryOnEarOffsets(
            vertical=row.get("offset_vertical") or 0,
            depth=row.get("offset_depth") or 0,
            lateral=row.get("offset_lateral") or 0,
        ),
        side_lateral_nudge=TryOnSideLateralNudge(
            left=row.get("nudge_left") or 0,
            right=row.get("nudge_right") or 0,
        ),
        jewelry_type=JewelryType(row.get("jewelry_type") or "drop"),
        version=int(row.get("config_version") or 1),
        updated_at=_ts_iso(float(row["updated_at"])),
    )


def config_to_row(
    cfg: ModelTryOnConfig,
    *,
    owner_user_id: str | None = None,
) -> dict[str, Any]:
    profile_x = profile_y = profile_z = None
    if cfg.profile_point is not None:
        profile_x = cfg.profile_point.x
        profile_y = cfg.profile_point.y
        profile_z = cfg.profile_point.z
    rot = cfg.jewelry_rotation or TryOnJewelryRotation()
    offsets = cfg.ear_offsets or TryOnEarOffsets()
    nudge = cfg.side_lateral_nudge or TryOnSideLateralNudge()
    return {
        "job_id": cfg.job_id,
        "hanger_x": cfg.hanger_point.x,
        "hanger_y": cfg.hanger_point.y,
        "hanger_z": cfg.hanger_point.z,
        "profile_x": profile_x,
        "profile_y": profile_y,
        "profile_z": profile_z,
        "spin_deg": rot.spin_deg,
        "pitch_deg": rot.pitch_deg,
        "roll_deg": rot.roll_deg,
        "api_vertical_flip": cfg.api_vertical_flip,
        "offset_vertical": offsets.vertical,
        "offset_depth": offsets.depth,
        "offset_lateral": offsets.lateral,
        "nudge_left": nudge.left,
        "nudge_right": nudge.right,
        "jewelry_type": cfg.jewelry_type.value,
        "config_version": cfg.version,
        "owner_user_id": owner_user_id,
    }


def merge_put(
    job_id: str,
    existing: ModelTryOnConfig | None,
    body: ModelTryOnConfigPut,
    *,
    require_hanger: bool,
) -> ModelTryOnConfig:
    if existing is None:
        if body.hanger_point is None:
            if require_hanger:
                raise ValueError("hanger_point is required when creating try-on config.")
            hanger = ModelPoint3D(x=0.0, y=0.5, z=0.0)
        else:
            hanger = clamp_point(body.hanger_point)
        profile = clamp_point(body.profile_point) if body.profile_point is not None else None
        rot = clamp_rotation(body.jewelry_rotation or TryOnJewelryRotation())
        cfg = ModelTryOnConfig(
            job_id=job_id,
            hanger_point=hanger,
            profile_point=profile,
            jewelry_rotation=rot,
            api_vertical_flip=bool(body.api_vertical_flip) if body.api_vertical_flip is not None else False,
            ear_offsets=body.ear_offsets or TryOnEarOffsets(),
            side_lateral_nudge=body.side_lateral_nudge or TryOnSideLateralNudge(),
            jewelry_type=body.jewelry_type or JewelryType.drop,
            version=1,
            updated_at=_ts_iso(time.time()),
        )
    else:
        hanger = (
            clamp_point(body.hanger_point)
            if body.hanger_point is not None
            else existing.hanger_point
        )
        if body.profile_point is not None:
            profile = clamp_point(body.profile_point)
        else:
            profile = existing.profile_point
        rot_in = body.jewelry_rotation
        rot = (
            clamp_rotation(rot_in)
            if rot_in is not None
            else (existing.jewelry_rotation or TryOnJewelryRotation())
        )
        cfg = ModelTryOnConfig(
            job_id=job_id,
            hanger_point=hanger,
            profile_point=profile,
            jewelry_rotation=rot,
            api_vertical_flip=(
                body.api_vertical_flip
                if body.api_vertical_flip is not None
                else existing.api_vertical_flip
            ),
            ear_offsets=body.ear_offsets or existing.ear_offsets or TryOnEarOffsets(),
            side_lateral_nudge=(
                body.side_lateral_nudge or existing.side_lateral_nudge or TryOnSideLateralNudge()
            ),
            jewelry_type=body.jewelry_type or existing.jewelry_type,
            version=existing.version,
            updated_at=_ts_iso(time.time()),
        )
    validate_profile_distance(cfg.hanger_point, cfg.profile_point)
    return cfg


def calibration_row_to_model(row: dict[str, Any]) -> UserTryOnCalibration:
    return UserTryOnCalibration(
        left_offset=TryOnEarOffsets(
            vertical=row.get("left_vertical") or 0,
            depth=row.get("left_depth") or 0,
            lateral=row.get("left_lateral") or 0,
        ),
        right_offset=TryOnEarOffsets(
            vertical=row.get("right_vertical") or 0,
            depth=row.get("right_depth") or 0,
            lateral=row.get("right_lateral") or 0,
        ),
        updated_at=_ts_iso(float(row["updated_at"])),
    )


class TryOnConfigStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        try_on_db.init_schema(db_path)

    def get(self, job_id: str) -> ModelTryOnConfig | None:
        row = try_on_db.get_model_try_on(self.db_path, job_id)
        if not row:
            return None
        return row_to_config(row)

    def get_owner(self, job_id: str) -> str | None:
        row = try_on_db.get_model_try_on(self.db_path, job_id)
        if not row:
            return None
        owner = row.get("owner_user_id")
        return str(owner) if owner else None

    def upsert(
        self,
        job_id: str,
        body: ModelTryOnConfigPut,
        *,
        require_hanger: bool = True,
        owner_user_id: str | None = None,
    ) -> ModelTryOnConfig:
        existing = self.get(job_id)
        cfg = merge_put(job_id, existing, body, require_hanger=require_hanger)
        row = config_to_row(cfg, owner_user_id=owner_user_id)
        if existing is not None:
            prev = try_on_db.get_model_try_on(self.db_path, job_id)
            if prev and prev.get("owner_user_id"):
                row["owner_user_id"] = prev["owner_user_id"]
        elif owner_user_id:
            row["owner_user_id"] = owner_user_id
        saved = try_on_db.upsert_model_try_on(self.db_path, row)
        return row_to_config(saved)

    def upsert_hanger_only(
        self,
        job_id: str,
        point: ModelPoint3D,
        *,
        owner_user_id: str | None = None,
    ) -> ModelTryOnConfig:
        return self.upsert(
            job_id,
            ModelTryOnConfigPut(hanger_point=point),
            require_hanger=True,
            owner_user_id=owner_user_id,
        )

    def delete_for_job(self, job_id: str) -> None:
        try_on_db.delete_model_try_on(self.db_path, job_id)

    def get_user_calibration(self, user_id: str) -> UserTryOnCalibration | None:
        row = try_on_db.get_user_calibration(self.db_path, user_id)
        if not row:
            return None
        return calibration_row_to_model(row)

    def upsert_user_calibration(
        self,
        user_id: str,
        body: UserTryOnCalibrationPut,
    ) -> UserTryOnCalibration:
        existing = self.get_user_calibration(user_id)
        left = body.left_offset or (existing.left_offset if existing else TryOnEarOffsets())
        right = body.right_offset or (existing.right_offset if existing else TryOnEarOffsets())
        row = try_on_db.upsert_user_calibration(
            self.db_path,
            user_id,
            {
                "left_vertical": left.vertical,
                "left_depth": left.depth,
                "left_lateral": left.lateral,
                "right_vertical": right.vertical,
                "right_depth": right.depth,
                "right_lateral": right.lateral,
            },
        )
        return calibration_row_to_model(row)
