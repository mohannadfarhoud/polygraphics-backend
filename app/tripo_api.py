"""REST routes for Tripo cloud image-to-3D generation."""

from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .tripo_models import TripoImageToModelResponse
from .tripo_service import TripoService

router = APIRouter(tags=["tripo"])

_service: TripoService | None = None

_IMAGE_MIME = frozenset(
    {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "image/bmp",
    }
)


def init_tripo_api(
    *,
    db_path: Path,
    root_dir: Path,
    download_url_for,
) -> None:
    global _service
    _service = TripoService(
        db_path=db_path,
        root_dir=root_dir,
        download_url_for=download_url_for,
    )


def get_tripo_service() -> TripoService:
    if _service is None:
        raise RuntimeError("tripo_api not initialized")
    return _service


@router.post(
    "/tripo/image-to-model",
    response_model=TripoImageToModelResponse,
    status_code=202,
    responses={
        202: {
            "description": (
                "Tripo job accepted; poll GET until completed, then download. "
                "Uses Tripo SDK when TRIPO_API_KEY is set; otherwise returns a placeholder GLB."
            )
        },
    },
)
async def post_tripo_image_to_model(
    image: UploadFile = File(..., description="Single product/object photo (JPEG, PNG, or WebP)."),
) -> JSONResponse:
    svc = get_tripo_service()
    if not svc.is_configured():
        raise HTTPException(
            status_code=503,
            detail="Tripo is not available (install tripo3d: pip install tripo3d).",
        )

    mime = (image.content_type or "").split(";")[0].strip().lower()
    if mime and mime not in _IMAGE_MIME:
        raise HTTPException(status_code=400, detail="image must be JPEG, PNG, WebP, or BMP")

    max_bytes = svc.max_upload_bytes()
    raw = await image.read()
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"image exceeds {max_bytes // (1024 * 1024)} MB limit",
        )
    if not raw:
        raise HTTPException(status_code=400, detail="image is empty")

    try:
        resp = svc.create_job(
            image_bytes=raw,
            filename=image.filename or "image.jpg",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from None

    return JSONResponse(status_code=202, content=resp.model_dump(mode="json"))


@router.get("/tripo/image-to-model/{tripo_job_id}", response_model=TripoImageToModelResponse)
def get_tripo_image_to_model(tripo_job_id: str) -> TripoImageToModelResponse:
    resp = get_tripo_service().get_job(tripo_job_id)
    if resp is None:
        raise HTTPException(status_code=404, detail="Tripo job not found")
    return resp


@router.get("/tripo/image-to-model/{tripo_job_id}/download")
def download_tripo_model(tripo_job_id: str) -> FileResponse:
    model_path = get_tripo_service().resolve_model_path(tripo_job_id)
    if model_path is None:
        raise HTTPException(status_code=404, detail="Model file not ready or not found")
    mt, _ = mimetypes.guess_type(str(model_path))
    return FileResponse(
        str(model_path),
        filename=f"{tripo_job_id}.glb",
        media_type=mt or "model/gltf-binary",
    )
