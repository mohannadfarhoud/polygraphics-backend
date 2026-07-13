"""REST routes for PicPolish isolation dataset, training, and inference."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response

from .auth_deps import get_current_user
from .auth_models import UserPublic
from .isolation_models import (
    IsolationDatasetCreate,
    IsolationDatasetDetail,
    IsolationDatasetSummary,
    IsolationHealthResponse,
    IsolationModelSummary,
    IsolationPairUploadResult,
    IsolationPredictJsonResponse,
    IsolationTrainRequest,
    IsolationTrainResponse,
)
from .isolation_quota import IsolationQuotaExceededError, IsolationQuotaStatus
from .isolation_service import IsolationService

router = APIRouter(tags=["isolation"])
log = logging.getLogger(__name__)

_service: IsolationService | None = None


def init_isolation_api(
    *,
    db_path: Path,
    root_dir: Path,
    asset_url_for,
) -> None:
    global _service
    _service = IsolationService(db_path=db_path, root_dir=root_dir, asset_url_for=asset_url_for)


def get_isolation_service() -> IsolationService:
    if _service is None:
        raise RuntimeError("isolation_api not initialized")
    return _service


def _current_user(user: UserPublic = Depends(get_current_user)) -> UserPublic:
    return user


@router.get("/isolation/health", response_model=IsolationHealthResponse)
def isolation_health() -> IsolationHealthResponse:
    return get_isolation_service().health()


@router.get("/isolation/quota", response_model=IsolationQuotaStatus)
def isolation_quota(user: UserPublic = Depends(_current_user)) -> IsolationQuotaStatus:
    return get_isolation_service().get_quota(user.user_id)


@router.post("/isolation/datasets", response_model=IsolationDatasetSummary, status_code=201)
def create_dataset(body: IsolationDatasetCreate) -> IsolationDatasetSummary:
    return get_isolation_service().create_dataset(name=body.name, owner_user_id=None)


@router.get("/isolation/datasets", response_model=list[IsolationDatasetSummary])
def list_datasets() -> list[IsolationDatasetSummary]:
    return get_isolation_service().list_datasets()


@router.get("/isolation/datasets/{dataset_id}", response_model=IsolationDatasetDetail)
def get_dataset(dataset_id: str) -> IsolationDatasetDetail:
    resp = get_isolation_service().get_dataset(dataset_id)
    if resp is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return resp


@router.delete("/isolation/datasets/{dataset_id}", status_code=204)
def delete_dataset(dataset_id: str) -> Response:
    if not get_isolation_service().delete_dataset(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found")
    return Response(status_code=204)


@router.post("/isolation/datasets/{dataset_id}/pairs", response_model=IsolationPairUploadResult)
async def upload_pairs(
    dataset_id: str,
    indices: str = Form(..., description="Comma-separated pair indices, e.g. 0,1,2"),
    before: list[UploadFile] = File(..., description="Before images (same order as indices)"),
    after: list[UploadFile] = File(..., description="After studio cutout images"),
    mask: list[UploadFile] | None = File(default=None, description="Optional explicit masks"),
) -> IsolationPairUploadResult:
    try:
        index_list = [int(x.strip()) for x in indices.split(",") if x.strip() != ""]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="indices must be comma-separated integers") from exc
    if len(index_list) != len(before) or len(index_list) != len(after):
        raise HTTPException(status_code=400, detail="indices, before, and after counts must match")
    if mask is not None and len(mask) not in (0, len(index_list)):
        raise HTTPException(status_code=400, detail="mask file count must be 0 or match indices")

    items: list[dict] = []
    for i, idx in enumerate(index_list):
        b_raw = await before[i].read()
        a_raw = await after[i].read()
        if not b_raw or not a_raw:
            raise HTTPException(status_code=400, detail=f"empty before/after at index {idx}")
        m_raw = None
        if mask and len(mask) > i and mask[i] is not None:
            m_raw = await mask[i].read()
        items.append(
            {
                "index": idx,
                "before_bytes": b_raw,
                "after_bytes": a_raw,
                "mask_bytes": m_raw,
                "before_name": before[i].filename or "before.jpg",
                "after_name": after[i].filename or "after.jpg",
            }
        )

    try:
        uploaded, pair_indices = get_isolation_service().add_pairs(dataset_id=dataset_id, items=items)
    except KeyError:
        raise HTTPException(status_code=404, detail="Dataset not found") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    return IsolationPairUploadResult(dataset_id=dataset_id, uploaded=uploaded, pair_indices=pair_indices)


@router.post("/isolation/train", response_model=IsolationTrainResponse, status_code=202)
def start_train(body: IsolationTrainRequest) -> JSONResponse:
    try:
        resp = get_isolation_service().start_train(
            dataset_id=body.dataset_id,
            base_model=body.base_model,
            epochs=body.epochs,
            val_split=body.val_split,
            force_min_pairs=body.force_min_pairs,
            owner_user_id=None,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Dataset not found") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return JSONResponse(status_code=202, content=resp.model_dump(mode="json"))


@router.get("/isolation/train/{job_id}", response_model=IsolationTrainResponse)
def get_train(job_id: str) -> IsolationTrainResponse:
    resp = get_isolation_service().get_train(job_id)
    if resp is None:
        raise HTTPException(status_code=404, detail="Train job not found")
    return resp


@router.get("/isolation/models", response_model=list[IsolationModelSummary])
def list_models() -> list[IsolationModelSummary]:
    return get_isolation_service().list_models()


@router.get("/isolation/models/active", response_model=IsolationModelSummary)
def get_active_model() -> IsolationModelSummary:
    resp = get_isolation_service().get_active_model()
    if resp is None:
        raise HTTPException(status_code=404, detail="No active isolation model")
    return resp


@router.post("/isolation/models/{model_id}/activate", response_model=IsolationModelSummary)
def activate_model(model_id: str) -> IsolationModelSummary:
    try:
        return get_isolation_service().activate_model(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Model not found") from None


@router.post("/isolation/models/import", response_model=IsolationModelSummary, status_code=201)
async def import_model(
    name: str = Form(...),
    file: UploadFile = File(..., description="model.onnx"),
    dataset_id: str | None = Form(default=None),
    activate: bool = Form(default=False),
) -> IsolationModelSummary:
    raw = await file.read()
    if len(raw) < 1024:
        raise HTTPException(status_code=400, detail="ONNX file too small")
    return get_isolation_service().import_model(
        name=name,
        onnx_bytes=raw,
        dataset_id=dataset_id,
        metrics={"imported": True},
        owner_user_id=None,
        activate=activate,
    )


@router.post("/isolation/predict")
async def predict_isolation(
    file: UploadFile = File(...),
    model_id: str | None = Form(default=None),
    response_format: str = Query(default="png", alias="format", description="png or json"),
    user: UserPublic = Depends(_current_user),
):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty image")
    return_json = response_format.strip().lower() == "json"
    try:
        png, body, _latency = get_isolation_service().predict(
            user_id=user.user_id,
            image_bytes=raw,
            model_id=model_id,
            return_json=return_json,
        )
    except IsolationQuotaExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from None
    except KeyError:
        raise HTTPException(status_code=404, detail="Model not found") from None
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    if return_json and body:
        return IsolationPredictJsonResponse(**body)
    assert png is not None
    return Response(content=png, media_type="image/png")
