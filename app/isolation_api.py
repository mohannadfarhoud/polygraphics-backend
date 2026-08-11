"""REST routes for PicPolish isolation dataset, training, and inference.

Training / dataset upload require any authenticated user.
Admin panel (statistics, pairs, activate/import, delete, dedupe, retrain-all) requires admin.
Predict + health + read-only model listing remain public.
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from .auth_deps import get_current_user, get_optional_current_user
from .auth_models import UserPublic
from .auth_service import GoogleAuthError, get_user_from_access_token
from .isolation_auth import ensure_trainer_user, is_admin_email, require_admin
from .isolation_models import (
    IsolationDatasetCreate,
    IsolationDatasetDetail,
    IsolationDatasetSummary,
    IsolationDedupeResult,
    IsolationHealthResponse,
    IsolationModelSummary,
    IsolationPairListResponse,
    IsolationPairUploadResult,
    IsolationPredictJsonResponse,
    IsolationRetrainAllRequest,
    IsolationRetrainAllResponse,
    IsolationStatistics,
    IsolationTrainRequest,
    IsolationTrainResponse,
)
from .isolation_quota import IsolationQuotaExceededError, IsolationQuotaStatus
from .isolation_service import IsolationService

router = APIRouter(tags=["isolation"])
log = logging.getLogger(__name__)

_service: IsolationService | None = None


def _require_admin_for_image(
    user: UserPublic | None = Depends(get_optional_current_user),
    access_token: str | None = Query(
        default=None,
        description="Optional JWT for <img src> (Bearer also works).",
    ),
) -> UserPublic:
    """Admin gate that also accepts ?access_token= for image tags."""
    if user is None and access_token:
        try:
            user = get_user_from_access_token(get_isolation_service().db_path, access_token.strip())
        except GoogleAuthError as exc:
            raise HTTPException(status_code=401, detail=str(exc)) from exc
    if user is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    if not is_admin_email(user.email):
        raise HTTPException(status_code=403, detail="Admin account required. Login as user 'admin'.")
    return user


def init_isolation_api(
    *,
    db_path: Path,
    root_dir: Path,
    asset_url_for,
) -> None:
    global _service
    _service = IsolationService(db_path=db_path, root_dir=root_dir, asset_url_for=asset_url_for)
    try:
        ensure_trainer_user(db_path)
    except Exception as exc:
        log.warning("could not ensure isolation trainer user: %s", exc)


def get_isolation_service() -> IsolationService:
    if _service is None:
        raise RuntimeError("isolation_api not initialized")
    return _service


@router.get("/isolation/health", response_model=IsolationHealthResponse)
def isolation_health() -> IsolationHealthResponse:
    return get_isolation_service().health()


@router.get("/isolation/quota", response_model=IsolationQuotaStatus)
def isolation_quota() -> IsolationQuotaStatus:
    return get_isolation_service().get_quota("anonymous")


@router.get("/isolation/statistics", response_model=IsolationStatistics)
def isolation_statistics(
    dataset_id: str | None = Query(default=None),
    user: UserPublic = Depends(require_admin),
) -> IsolationStatistics:
    del user
    try:
        return get_isolation_service().get_statistics(dataset_id=dataset_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Dataset not found") from None


@router.get("/isolation/pairs", response_model=IsolationPairListResponse)
def list_uploaded_pairs(
    dataset_id: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    user: UserPublic = Depends(require_admin),
) -> IsolationPairListResponse:
    """Admin table: uploaded before/after pairs with uploader + timestamp."""
    del user
    try:
        return get_isolation_service().list_uploaded_pairs(
            dataset_id=dataset_id,
            limit=limit,
            offset=offset,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Dataset not found") from None


@router.get("/isolation/pairs/{pair_id}/before")
def get_pair_before(
    pair_id: str,
    _admin: UserPublic = Depends(_require_admin_for_image),
) -> FileResponse:
    return _pair_image_response(pair_id, "before")


@router.get("/isolation/pairs/{pair_id}/after")
def get_pair_after(
    pair_id: str,
    _admin: UserPublic = Depends(_require_admin_for_image),
) -> FileResponse:
    return _pair_image_response(pair_id, "after")


@router.get("/isolation/pairs/{pair_id}/mask")
def get_pair_mask(
    pair_id: str,
    _admin: UserPublic = Depends(_require_admin_for_image),
) -> FileResponse:
    return _pair_image_response(pair_id, "mask")


def _pair_image_response(pair_id: str, kind: str) -> FileResponse:
    try:
        path = get_isolation_service().resolve_pair_image(pair_id=pair_id, kind=kind)
    except KeyError:
        raise HTTPException(status_code=404, detail="Pair not found") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    media_type, _ = mimetypes.guess_type(str(path))
    return FileResponse(str(path), media_type=media_type or "application/octet-stream", filename=path.name)


@router.post("/isolation/datasets", response_model=IsolationDatasetSummary, status_code=201)
def create_dataset(
    body: IsolationDatasetCreate,
    user: UserPublic = Depends(get_current_user),
) -> IsolationDatasetSummary:
    return get_isolation_service().create_dataset(name=body.name, owner_user_id=user.user_id)


@router.get("/isolation/datasets", response_model=list[IsolationDatasetSummary])
def list_datasets(user: UserPublic = Depends(get_current_user)) -> list[IsolationDatasetSummary]:
    return get_isolation_service().list_datasets()


@router.get("/isolation/datasets/{dataset_id}", response_model=IsolationDatasetDetail)
def get_dataset(
    dataset_id: str,
    user: UserPublic = Depends(get_current_user),
) -> IsolationDatasetDetail:
    resp = get_isolation_service().get_dataset(dataset_id)
    if resp is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return resp


@router.delete("/isolation/datasets/{dataset_id}", status_code=204)
def delete_dataset(
    dataset_id: str,
    user: UserPublic = Depends(require_admin),
) -> Response:
    if not get_isolation_service().delete_dataset(dataset_id):
        raise HTTPException(status_code=404, detail="Dataset not found")
    return Response(status_code=204)


@router.post("/isolation/datasets/{dataset_id}/pairs", response_model=IsolationPairUploadResult)
async def upload_pairs(
    dataset_id: str,
    before: list[UploadFile] = File(..., description="Before image(s). One file = a single couple."),
    after: list[UploadFile] = File(..., description="After cutout image(s), same order as before"),
    mask: list[UploadFile] | None = File(default=None, description="Optional mask(s)"),
    indices: str | None = Form(
        default=None,
        description="Optional comma-separated indices. Omit for a single couple (auto next index).",
    ),
    user: UserPublic = Depends(get_current_user),
) -> IsolationPairUploadResult:
    submitted_photos = len(before) + len(after)
    submitted_pairs = max(len(before), len(after))

    def record_failure(detail: str) -> None:
        get_isolation_service().record_upload_event(
            dataset_id=dataset_id,
            user_id=user.user_id,
            submitted_photos=submitted_photos,
            submitted_pairs=submitted_pairs,
            successful_pairs=0,
            error=detail,
        )

    if not before or not after:
        record_failure("before and after are required")
        raise HTTPException(status_code=400, detail="before and after are required")
    if len(before) != len(after):
        record_failure("before and after counts must match")
        raise HTTPException(status_code=400, detail="before and after counts must match")
    if mask is not None and len(mask) not in (0, len(before)):
        record_failure("mask file count must be 0 or match before/after")
        raise HTTPException(status_code=400, detail="mask file count must be 0 or match before/after")

    index_list: list[int | None]
    if indices is None or not str(indices).strip():
        # Single couple or batch without indices → auto-assign sequential next indices.
        index_list = [None] * len(before)
    else:
        try:
            index_list = [int(x.strip()) for x in str(indices).split(",") if x.strip() != ""]
        except ValueError as exc:
            record_failure("indices must be comma-separated integers")
            raise HTTPException(status_code=400, detail="indices must be comma-separated integers") from exc
        if len(index_list) != len(before):
            record_failure("indices count must match before/after")
            raise HTTPException(status_code=400, detail="indices count must match before/after")

    items: list[dict] = []
    for i, idx in enumerate(index_list):
        b_raw = await before[i].read()
        a_raw = await after[i].read()
        if not b_raw or not a_raw:
            record_failure(f"empty before/after at pair {i}")
            raise HTTPException(status_code=400, detail=f"empty before/after at pair {i}")
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
        uploaded, pair_indices, rejected_duplicates = get_isolation_service().add_pairs(
            dataset_id=dataset_id,
            items=items,
            uploaded_by_user_id=user.user_id,
        )
    except KeyError:
        record_failure("Dataset not found")
        raise HTTPException(status_code=404, detail="Dataset not found") from None
    except ValueError as exc:
        record_failure(str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except Exception as exc:
        record_failure(str(exc))
        raise

    get_isolation_service().record_upload_event(
        dataset_id=dataset_id,
        user_id=user.user_id,
        submitted_photos=submitted_photos,
        submitted_pairs=submitted_pairs,
        successful_pairs=uploaded,
        error=(
            f"Rejected {rejected_duplicates} duplicate pair(s)"
            if rejected_duplicates and uploaded
            else None
        ),
    )
    return IsolationPairUploadResult(
        dataset_id=dataset_id,
        uploaded=uploaded,
        pair_indices=pair_indices,
        rejected_duplicates=rejected_duplicates,
    )


@router.post("/isolation/train", response_model=IsolationTrainResponse, status_code=202)
def start_train(
    body: IsolationTrainRequest,
    user: UserPublic = Depends(get_current_user),
) -> JSONResponse:
    try:
        resp = get_isolation_service().start_train(
            dataset_id=body.dataset_id,
            base_model=body.base_model,
            epochs=body.epochs,
            val_split=body.val_split,
            force_min_pairs=body.force_min_pairs,
            owner_user_id=user.user_id,
            grow_active=body.grow_active,
            resume_from_model_id=body.resume_from_model_id,
            auto_activate=body.auto_activate,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Dataset not found") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return JSONResponse(status_code=202, content=resp.model_dump(mode="json"))


@router.post(
    "/isolation/datasets/{dataset_id}/dedupe",
    response_model=IsolationDedupeResult,
    operation_id="isolationDedupeDataset",
    summary="Admin: remove duplicate couples",
)
def dedupe_dataset(
    dataset_id: str,
    user: UserPublic = Depends(require_admin),
) -> IsolationDedupeResult:
    """Admin: remove exact duplicate before/after couples (keeps lowest index)."""
    del user
    try:
        return get_isolation_service().dedupe_dataset(dataset_id, delete_files=True)
    except KeyError:
        raise HTTPException(status_code=404, detail="Dataset not found") from None


@router.post(
    "/isolation/retrain-all",
    response_model=IsolationRetrainAllResponse,
    status_code=202,
    operation_id="isolationRetrainAll",
    summary="Admin: purge duplicates and retrain from current images",
)
def retrain_all(
    body: IsolationRetrainAllRequest = Body(default=IsolationRetrainAllRequest()),
    user: UserPublic = Depends(require_admin),
) -> JSONResponse:
    """Admin: purge duplicate couples then queue a full retrain of the current dataset.

    UI: login as admin, then POST `{}` (or `{ "dataset_id": "..." }`).
    Default: `purge_duplicates=true`, `grow_active=false` (rebuild, do not continue the old model).
    """
    try:
        resp = get_isolation_service().retrain_all(
            dataset_id=body.dataset_id,
            base_model=body.base_model,
            epochs=body.epochs,
            val_split=body.val_split,
            force_min_pairs=body.force_min_pairs,
            grow_active=body.grow_active,
            auto_activate=body.auto_activate,
            purge_duplicates=body.purge_duplicates,
            owner_user_id=user.user_id,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="Dataset not found") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return JSONResponse(status_code=202, content=resp.model_dump(mode="json"))


@router.get("/isolation/train/{job_id}", response_model=IsolationTrainResponse)
def get_train(
    job_id: str,
    user: UserPublic = Depends(get_current_user),
) -> IsolationTrainResponse:
    del user
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
def activate_model(
    model_id: str,
    user: UserPublic = Depends(require_admin),
) -> IsolationModelSummary:
    del user
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
    user: UserPublic = Depends(require_admin),
) -> IsolationModelSummary:
    raw = await file.read()
    if len(raw) < 1024:
        raise HTTPException(status_code=400, detail="ONNX file too small")
    return get_isolation_service().import_model(
        name=name,
        onnx_bytes=raw,
        dataset_id=dataset_id,
        metrics={"imported": True},
        owner_user_id=user.user_id,
        activate=activate,
    )


@router.post("/isolation/predict")
async def predict_isolation(
    file: UploadFile = File(...),
    model_id: str | None = Form(default=None),
    response_format: str = Query(default="png", alias="format", description="png or json"),
):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty image")
    return_json = response_format.strip().lower() == "json"
    try:
        png, body, _latency = get_isolation_service().predict(
            user_id=None,
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
