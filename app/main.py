from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from starlette.staticfiles import StaticFiles

from .config import PipelineConfig
from .interfaces import JobRepository, JobStatus, NoopWebSocketNotifier, WebSocketNotifier
from .job_manager import JobManager, JobRecord, ModelListItem
from .pipeline import ReconstructionPipeline
from .reconstruction import Dust3RReconstructor
from .segmentation import SamSegmenter
from .runtime_settings import RuntimeSettings, SettingsStore
from .server_status import collect_server_status


ROOT_DIR = Path(os.getenv("APP_ROOT_DIR", Path(__file__).resolve().parents[1])).resolve()
UPLOAD_DIR = ROOT_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="polyGraphics 3D Backend",
    description="API for 2D-to-3D reconstruction pipeline (SAM + DUSt3R + Open3D)",
    version="1.0.0",
    docs_url="/swagger",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)
_settings_cors = os.getenv("APP_CORS_ORIGINS", "*").strip()
_origins = [o.strip() for o in _settings_cors.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins if _origins != ["*"] else ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
settings_store = SettingsStore(ROOT_DIR)


def _effective_model_base_url(settings: RuntimeSettings) -> str:
    """Prefix used in job.model_url; files are served at GET /output/{{job_id}}.glb."""
    for key in ("APP_MODEL_BASE_URL", "APP_CDN_BASE_URL"):
        v = os.getenv(key, "").strip()
        if v:
            return v.rstrip("/")
    base = (settings.cdn_base_url or "").strip().rstrip("/")
    if not base or "cdn.yoursite.com" in base:
        return "http://127.0.0.1:8000/output"
    return base


def _build_pipeline_for_job(
    settings: RuntimeSettings,
    job_repo: JobRepository,
    notifier: WebSocketNotifier | None,
) -> ReconstructionPipeline:
    _ = notifier
    return ReconstructionPipeline(
        config=PipelineConfig(
            root_dir=ROOT_DIR,
            output_dir_name=settings.output_dir_name,
            masked_dir_name=settings.masked_dir_name,
            nb_neighbors=settings.nb_neighbors,
            std_ratio=settings.std_ratio,
            poisson_depth=settings.poisson_depth,
            poisson_density_quantile=settings.poisson_density_quantile,
            decimation_target_triangles=settings.decimation_target_triangles,
            cdn_base_url=_effective_model_base_url(settings),
        ),
        runtime_settings=settings,
        segmenter=SamSegmenter(settings),
        reconstructor=Dust3RReconstructor(settings),
        job_repo=job_repo,
        notifier=NoopWebSocketNotifier(),
    )


job_manager = JobManager(
    root_dir=ROOT_DIR,
    settings_store=settings_store,
    build_pipeline=_build_pipeline_for_job,
    notifier=None,
)


def _assert_upload_allowed(job_id: str) -> None:
    existing = job_manager.get_job(job_id)
    if existing and existing.status in (JobStatus.PROCESSING, JobStatus.QUEUED):
        raise HTTPException(
            status_code=409,
            detail=f"Job is busy ({existing.status.value}); wait or use stop before uploading again.",
        )


async def _save_job_files(job_id: str, files: list[UploadFile]) -> None:
    job_upload_dir = UPLOAD_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)
    for p in job_upload_dir.glob("input_*"):
        if p.is_file():
            p.unlink()
    for idx, upload in enumerate(files):
        suffix = Path(upload.filename or "").suffix or ".png"
        image_path = job_upload_dir / f"input_{idx:03d}{suffix}"
        with image_path.open("wb") as f:
            f.write(await upload.read())


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/swagger")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/settings", response_model=RuntimeSettings)
def get_settings() -> RuntimeSettings:
    return settings_store.load()


@app.put("/settings", response_model=RuntimeSettings)
def update_settings(payload: RuntimeSettings) -> RuntimeSettings:
    return settings_store.save(payload)


@app.get("/server/status")
def server_status() -> dict:
    return collect_server_status(ROOT_DIR)


@app.get("/jobs", response_model=list[JobRecord])
def list_jobs() -> list[JobRecord]:
    return job_manager.list_jobs()


@app.get("/jobs/{job_id}", response_model=JobRecord)
def get_job(job_id: str) -> JobRecord:
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/jobs/{job_id}/stop", response_model=JobRecord)
def stop_job(job_id: str) -> JobRecord:
    try:
        return job_manager.request_stop(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found") from None


@app.post("/jobs/{job_id}/continue", response_model=JobRecord)
def continue_job(job_id: str) -> JobRecord:
    try:
        return job_manager.continue_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found") from None
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@app.post("/jobs/{job_id}/reprocess", response_model=JobRecord)
def reprocess_job(job_id: str) -> JobRecord:
    try:
        return job_manager.reprocess_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found") from None
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@app.get("/models", response_model=list[ModelListItem])
def list_models() -> list[ModelListItem]:
    settings = settings_store.load()
    output_dir = ROOT_DIR / settings.output_dir_name
    return job_manager.list_models(output_dir)


@app.post("/jobs", response_model=JobRecord)
async def create_job_from_uploads(
    files: list[UploadFile] = File(..., description="One or more images; job stays PENDING until you call POST /jobs/{job_id}/start"),
    job_id: str | None = Form(default=None),
) -> JobRecord:
    """Accept multipart images in the body, save them under uploads/{job_id}/, create job as PENDING (does not run pipeline yet)."""
    current_settings = settings_store.load()
    if len(files) > current_settings.max_images:
        raise HTTPException(status_code=400, detail=f"Too many files. max_images={current_settings.max_images}")
    if len(files) < 1:
        raise HTTPException(status_code=400, detail="At least one file is required")

    use_job_id = job_id or str(uuid.uuid4())
    _assert_upload_allowed(use_job_id)
    await _save_job_files(use_job_id, files)
    return job_manager.create_job_pending(use_job_id, len(files))


@app.post("/jobs/{job_id}/start", response_model=JobRecord)
def start_job(job_id: str) -> JobRecord:
    """Begin processing for a PENDING job (requires at least 2 images on disk)."""
    try:
        return job_manager.start_job(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found") from None
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@app.post("/jobs/reconstruct", response_model=JobRecord)
async def reconstruct(
    files: list[UploadFile] = File(...),
    job_id: str | None = Form(default=None),
) -> JobRecord:
    """Convenience: same as POST /jobs then POST /jobs/{id}/start — upload and run immediately."""
    current_settings = settings_store.load()
    if len(files) > current_settings.max_images:
        raise HTTPException(status_code=400, detail=f"Too many files. max_images={current_settings.max_images}")
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 images for reconstruction")

    use_job_id = job_id or str(uuid.uuid4())
    _assert_upload_allowed(use_job_id)
    await _save_job_files(use_job_id, files)
    try:
        return job_manager.enqueue_new_job(use_job_id, len(files))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


def _mount_output_static() -> None:
    s = settings_store.load()
    out = ROOT_DIR / s.output_dir_name
    out.mkdir(parents=True, exist_ok=True)
    app.mount("/output", StaticFiles(directory=str(out)), name="output_glb")


_mount_output_static()
