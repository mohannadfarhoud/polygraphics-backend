"""REST routes for per-model virtual try-on configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException

from .job_manager import JobManager
from .runtime_settings import RuntimeSettings, SettingsStore
from .try_on_config import (
    HangerPointBody,
    ModelPoint3D,
    ModelTryOnConfig,
    ModelTryOnConfigPut,
    TryOnConfigStore,
    UserTryOnCalibration,
    UserTryOnCalibrationPut,
)

router = APIRouter(tags=["try-on"])

_store: TryOnConfigStore | None = None
_job_manager: JobManager | None = None
_settings_store: SettingsStore | None = None
_root_dir: Path | None = None


def init_try_on_api(
    *,
    job_manager: JobManager,
    settings_store: SettingsStore,
    root_dir: Path,
) -> None:
    global _store, _job_manager, _settings_store, _root_dir
    _job_manager = job_manager
    _settings_store = settings_store
    _root_dir = root_dir
    _store = TryOnConfigStore(job_manager.db_path)


def get_try_on_store() -> TryOnConfigStore:
    if _store is None:
        raise RuntimeError("try_on_api not initialized")
    return _store


def _output_dir() -> Path:
    assert _settings_store is not None and _root_dir is not None
    settings = _settings_store.load()
    return _root_dir / settings.output_dir_name


def _model_output_exists(job_id: str) -> bool:
    out = _output_dir()
    if not out.is_dir():
        return False
    return (out / f"{job_id}.glb").is_file() or (out / f"{job_id}.ply").is_file()


def _assert_model_job_exists(job_id: str) -> None:
    assert _job_manager is not None
    job = _job_manager.get_job(job_id)
    if job is None and not _model_output_exists(job_id):
        raise HTTPException(status_code=404, detail="Job not found")


def _optional_user_id(
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
) -> str | None:
    v = (x_user_id or "").strip()
    return v or None


def _enforce_owner_access(job_id: str, user_id: str | None) -> None:
    """When a row has owner_user_id, require matching X-User-Id (403)."""
    store = get_try_on_store()
    owner = store.get_owner(job_id)
    if owner and owner != user_id:
        raise HTTPException(status_code=403, detail="Not authorized for this model")


def _resolve_owner_on_write(job_id: str, user_id: str | None) -> str | None:
    store = get_try_on_store()
    existing_owner = store.get_owner(job_id)
    if existing_owner:
        return existing_owner
    if user_id and os.getenv("APP_TRY_ON_BIND_OWNER", "1").strip().lower() not in ("0", "false", "no"):
        return user_id
    return None


@router.get("/models/{job_id}/try-on", response_model=ModelTryOnConfig)
def get_model_try_on(job_id: str, user_id: str | None = Depends(_optional_user_id)) -> ModelTryOnConfig:
    _assert_model_job_exists(job_id)
    _enforce_owner_access(job_id, user_id)
    cfg = get_try_on_store().get(job_id)
    if cfg is None:
        raise HTTPException(status_code=404, detail="Try-on config not found for this model")
    return cfg


@router.put("/models/{job_id}/try-on", response_model=ModelTryOnConfig)
def put_model_try_on(
    job_id: str,
    body: ModelTryOnConfigPut,
    user_id: str | None = Depends(_optional_user_id),
) -> ModelTryOnConfig:
    _assert_model_job_exists(job_id)
    _enforce_owner_access(job_id, user_id)
    store = get_try_on_store()
    existing = store.get(job_id)
    if existing is None and body.hanger_point is None:
        raise HTTPException(
            status_code=400,
            detail="hanger_point is required when creating try-on config.",
        )
    try:
        owner = _resolve_owner_on_write(job_id, user_id)
        return store.upsert(job_id, body, require_hanger=existing is None, owner_user_id=owner)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@router.get("/models/{job_id}/hanger-point", response_model=ModelPoint3D)
def get_hanger_point(job_id: str, user_id: str | None = Depends(_optional_user_id)) -> ModelPoint3D:
    _assert_model_job_exists(job_id)
    _enforce_owner_access(job_id, user_id)
    cfg = get_try_on_store().get(job_id)
    if cfg is None:
        raise HTTPException(status_code=404, detail="Hanger point not found for this model")
    return cfg.hanger_point


@router.put("/models/{job_id}/hanger-point", response_model=ModelPoint3D)
def put_hanger_point(
    job_id: str,
    body: HangerPointBody,
    user_id: str | None = Depends(_optional_user_id),
) -> ModelPoint3D:
    _assert_model_job_exists(job_id)
    _enforce_owner_access(job_id, user_id)
    store = get_try_on_store()
    try:
        owner = _resolve_owner_on_write(job_id, user_id)
        cfg = store.upsert_hanger_only(
            job_id,
            ModelPoint3D(x=body.x, y=body.y, z=body.z),
            owner_user_id=owner,
        )
        return cfg.hanger_point
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@router.get("/users/me/try-on-calibration", response_model=UserTryOnCalibration)
def get_user_try_on_calibration(
    user_id: str | None = Depends(_optional_user_id),
) -> UserTryOnCalibration:
    if not user_id:
        raise HTTPException(
            status_code=400,
            detail="X-User-Id header is required for user try-on calibration.",
        )
    cfg = get_try_on_store().get_user_calibration(user_id)
    if cfg is None:
        raise HTTPException(status_code=404, detail="Try-on calibration not found for this user")
    return cfg


@router.put("/users/me/try-on-calibration", response_model=UserTryOnCalibration)
def put_user_try_on_calibration(
    body: UserTryOnCalibrationPut,
    user_id: str | None = Depends(_optional_user_id),
) -> UserTryOnCalibration:
    if not user_id:
        raise HTTPException(
            status_code=400,
            detail="X-User-Id header is required for user try-on calibration.",
        )
    return get_try_on_store().upsert_user_calibration(user_id, body)
