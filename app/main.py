from __future__ import annotations

import html as html_module
import json
import mimetypes
import os
import uuid
from pathlib import Path
from typing import Annotated, Any

try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]

    _ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
    if _ENV_PATH.is_file():
        load_dotenv(_ENV_PATH, override=False)
except ImportError:
    pass

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import FileResponse, RedirectResponse, Response
from pydantic import BaseModel, Field
from starlette.responses import HTMLResponse, JSONResponse
from starlette.websockets import WebSocketDisconnect

from .config import PipelineConfig
from .interfaces import JobRepository, JobStatus, NoopWebSocketNotifier, WebSocketNotifier
from .job_manager import JobManager, JobRecord, ModelListItem, remote_workers_enabled
from .pipeline import ReconstructionPipeline
from .reconstruction import Dust3RReconstructor
from .segmentation import SamSegmenter
from .runtime_settings import RuntimeSettings, SettingsStore
from .settings_guide import settings_deployment_guide
from .server_status import collect_server_status
from .worker_hub import WorkerHub, init_hub

# Resolve project root reliably when running as a Windows service (CWD may be
# System32 or arbitrary). Relative APP_ROOT_DIR is anchored to the repo directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_raw_root = (os.getenv("APP_ROOT_DIR") or "").strip()
if not _raw_root:
    ROOT_DIR = _REPO_ROOT.resolve()
elif Path(_raw_root).is_absolute():
    ROOT_DIR = Path(_raw_root).resolve()
else:
    ROOT_DIR = (_REPO_ROOT / _raw_root).resolve()

UPLOAD_DIR = ROOT_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_root_path = os.getenv("APP_ROOT_PATH", "").strip()
if _root_path and not _root_path.startswith("/"):
    _root_path = "/" + _root_path
_root_path = _root_path.rstrip("/")

app = FastAPI(
    title="polyGraphics 3D Backend",
    description="API for 2D-to-3D reconstruction pipeline (SAM + MapAnything + Open3D)",
    version="1.0.0",
    # Custom /swagger + /redoc below: FastAPI defaults use scope root_path only; we do not set
    # FastAPI(root_path=…) because it breaks bare /output and /uploads behind strip-prefix proxies.
    # When APP_ROOT_PATH is set (public URL prefix), Swagger must fetch openapi.json under that prefix.
    docs_url=None,
    redoc_url=None,
    openapi_url="/openapi.json",
)


def custom_openapi() -> dict:
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi

    servers: list[dict] = [{"url": "/", "description": "Direct Uvicorn (e.g. http://127.0.0.1:8000)"}]
    if _root_path:
        servers.insert(
            0,
            {"url": _root_path, "description": "Public URL path prefix (same value as APP_ROOT_PATH)"},
        )
    app.openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version="3.1.0",
        description=app.description,
        routes=app.routes,
        servers=servers,
    )
    return app.openapi_schema


app.openapi = custom_openapi  # type: ignore[method-assign]


def _browser_openapi_url() -> str:
    """Browser-visible path for OpenAPI JSON when behind a strip-prefix proxy (APP_ROOT_PATH)."""
    return f"{_root_path}/openapi.json" if _root_path else "/openapi.json"


@app.get("/swagger", include_in_schema=False)
async def swagger_ui_html_route() -> HTMLResponse:
    return get_swagger_ui_html(
        openapi_url=_browser_openapi_url(),
        title=f"{app.title} - Swagger UI",
        oauth2_redirect_url=None,
        init_oauth=app.swagger_ui_init_oauth,
        swagger_ui_parameters=app.swagger_ui_parameters,
    )


@app.get("/redoc", include_in_schema=False)
async def redoc_html_route() -> HTMLResponse:
    return get_redoc_html(openapi_url=_browser_openapi_url(), title=f"{app.title} - ReDoc")


if _root_path:

    @app.get(f"{_root_path}/openapi.json", include_in_schema=False)
    async def openapi_json_public_prefix_mirror() -> JSONResponse:
        return JSONResponse(app.openapi())


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


@app.on_event("startup")
async def _startup_worker_hub() -> None:
    import asyncio
    import logging

    hub = WorkerHub()
    hub.set_loop(asyncio.get_running_loop())
    init_hub(hub)
    app.state.worker_hub = hub

    log = logging.getLogger("uvicorn.error")
    ws_extra = f", {_root_path}/internal/worker/ws" if _root_path else ""
    log.info(
        "Worker WebSocket endpoints registered: /internal/worker/ws%s (APP_ROOT_PATH=%r)",
        ws_extra,
        _root_path or "",
    )
    if remote_workers_enabled() and not _root_path:
        log.warning(
            "APP_ROOT_PATH is empty but remote workers are enabled. If clients use "
            "https://host/polygraph/..., set APP_ROOT_PATH=/polygraph on this API, "
            "or configure nginx to strip /polygraph before forwarding so only "
            "/internal/worker/ws is needed.",
        )


_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "0.0.0.0", "::1")


def _is_loopback(url: str) -> bool:
    u = url.lower()
    return any(h in u for h in _LOOPBACK_HOSTS)


def _relative_base(suffix: str) -> str:
    """Path-only base for models and uploads.

    When ``APP_ROOT_PATH`` is set (e.g. ``/polygraph``), assets are served at
    ``{APP_ROOT_PATH}/output`` and ``{APP_ROOT_PATH}/uploads`` on this same app
    so ``http://127.0.0.1:8000/polygraph/output/...`` works without a reverse proxy.
    """
    suffix = "/" + suffix.strip("/")
    rp = _root_path.rstrip("/") if _root_path else ""
    return f"{rp}{suffix}" if rp else suffix


def _relative_api_path(subpath: str) -> str:
    """URL path for FastAPI routes when ``APP_ROOT_PATH`` prefixes public URLs."""
    subpath = "/" + subpath.strip("/")
    rp = _root_path.rstrip("/") if _root_path else ""
    return f"{rp}{subpath}" if rp else subpath


def _effective_model_base_url(settings: RuntimeSettings) -> str:
    """Prefix used in job.model_url. Defaults to ``/output`` (relative). Set
    APP_MODEL_BASE_URL to a non-loopback absolute URL (e.g. a CDN) to override."""
    for key in ("APP_MODEL_BASE_URL", "APP_CDN_BASE_URL"):
        v = os.getenv(key, "").strip()
        if v and not _is_loopback(v):
            return v.rstrip("/")
    base = (settings.cdn_base_url or "").strip().rstrip("/")
    if base and "cdn.yoursite.com" not in base and not _is_loopback(base):
        return base
    return _relative_base("output")


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _uploads_base_url() -> str:
    """Public URL prefix for files under uploads/. Mirrors model base URL logic;
    relative by default so a frontend on the same proxy resolves it correctly."""
    v = os.getenv("APP_UPLOADS_BASE_URL", "").strip()
    if v and not _is_loopback(v):
        return v.rstrip("/")
    base = _effective_model_base_url(settings_store.load())
    if base.endswith("/output"):
        return base[: -len("/output")] + "/uploads"
    return _relative_base("uploads")


def _image_sample_url(job_id: str) -> str | None:
    job_dir = UPLOAD_DIR / job_id
    if not job_dir.is_dir():
        return None
    for p in sorted(job_dir.glob("input_*")):
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
            return f"{_uploads_base_url()}/{job_id}/{p.name}"
    return None


def _masked_view_urls(job_id: str) -> list[str]:
    """Public URLs for mirrored SAM/precut RGB under uploads/{job_id}/masked_views/."""
    d = UPLOAD_DIR / job_id / "masked_views"
    if not d.is_dir():
        return []
    base = _uploads_base_url().rstrip("/")
    out: list[str] = []
    for p in sorted(d.iterdir()):
        if (
            p.is_file()
            and p.name.startswith("masked_")
            and p.suffix.lower() in _IMAGE_EXTS
        ):
            out.append(f"{base}/{job_id}/masked_views/{p.name}")
    return out


def _should_rewrite_model_url(url: str | None) -> bool:
    if not url:
        return True
    u = url.strip()
    if _is_loopback(u):
        return True
    # Stale relative paths that pointed at the dev server
    if u.startswith("/") and "127.0.0.1" in u:
        return True
    return False


def _decorate_job_response(job: JobRecord) -> JobRecord:
    """Attach image_sample_url; rewrite stale loopback model_url from DB; backfill old rows."""
    settings = settings_store.load()
    job.image_sample_url = _image_sample_url(job.job_id)
    job.masked_view_urls = _masked_view_urls(job.job_id)
    job.masked_preview_page_url = _relative_api_path(f"jobs/{job.job_id}/masked-preview")

    if (
        job.model_format
        and job.status == JobStatus.COMPLETED
        and _should_rewrite_model_url(job.model_url)
    ):
        ext = job.model_format.lower().lstrip(".")
        if ext in ("glb", "ply"):
            base = _effective_model_base_url(settings)
            job.model_url = f"{base.rstrip('/')}/{job.job_id}.{ext}"

    # Older jobs finished before progress/stage columns were written reliably
    if job.status == JobStatus.COMPLETED:
        if job.progress is None:
            job.progress = 100
        if job.stage is None:
            job.stage = "completed"

    return job


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
            masks_dir_name=settings.masks_dir_name,
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


_JOB_STAGES_JSON = _REPO_ROOT / "ui" / "job-stages-progress.json"
_CAPTURE_GUIDE_JSON = _REPO_ROOT / "ui" / "capture-guide.json"


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    sp = f"{_root_path}/swagger" if _root_path else "/swagger"
    return RedirectResponse(url=sp)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


class WorkerFailBody(BaseModel):
    error: str


class WorkerProgressBody(BaseModel):
    stage: str = "processing"
    progress: int = Field(default=0, ge=0, le=100)


def verify_worker_token(x_worker_token: str | None = Header(default=None, alias="X-Worker-Token")) -> None:
    secret = os.getenv("APP_WORKER_TOKEN", "").strip()
    if not secret:
        raise HTTPException(status_code=503, detail="APP_WORKER_TOKEN is not set on the API server")
    if x_worker_token != secret:
        raise HTTPException(status_code=403, detail="Invalid worker token")


_MASKED_VIEW_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})


def _safe_masked_view_basename(raw_name: str) -> str | None:
    """Reject path traversal; only ``masked_*`` image names."""
    base = Path(raw_name).name
    if base != raw_name.strip():
        return None
    if not base.startswith("masked_"):
        return None
    suf = Path(base).suffix.lower()
    if suf not in _MASKED_VIEW_SUFFIXES:
        return None
    return base


@app.get(
    "/internal/worker/next",
    dependencies=[Depends(verify_worker_token)],
    response_model=None,
)
def internal_worker_next() -> dict[str, Any] | Response:
    """GPU worker long-polls (or polls) for the next ``QUEUED`` job. Requires ``APP_REMOTE_WORKERS=true``."""
    if not remote_workers_enabled():
        raise HTTPException(
            status_code=503,
            detail="Remote workers disabled. Set APP_REMOTE_WORKERS=true on the API server.",
        )
    payload = job_manager.claim_next_remote_job()
    if payload is None:
        return Response(status_code=204)
    return payload


@app.post("/internal/worker/jobs/{job_id}/complete", dependencies=[Depends(verify_worker_token)])
async def internal_worker_complete(
    job_id: str,
    model_format: str = Form(...),
    file: UploadFile = File(...),
) -> JobRecord:
    body = await file.read()
    try:
        return job_manager.complete_remote_job(job_id, body, model_format)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found") from None
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@app.post("/internal/worker/jobs/{job_id}/comparison-glb", dependencies=[Depends(verify_worker_token)])
async def internal_worker_comparison_glb(
    job_id: str,
    variant: str = Form(..., description="comparison GLB variant: mesh (MapAnything preview)"),
    file: UploadFile = File(...),
) -> dict[str, str]:
    """Sidecar mesh from ``compare_mesh_preview_with_gs``; call before ``/complete`` while the job is PROCESSING."""
    body = await file.read()
    try:
        return job_manager.save_remote_comparison_glb(job_id, variant, body)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found") from None
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@app.post("/internal/worker/jobs/{job_id}/masked-views", dependencies=[Depends(verify_worker_token)])
async def internal_worker_masked_views(
    job_id: str,
    files: Annotated[list[UploadFile], File()],
) -> dict[str, Any]:
    """Persist SAM/precut isolated views under ``uploads/{job_id}/masked_views/`` (remote worker split deploy).

    Multipart field name ``files`` — each part filename ``masked_*.png`` (or .jpg/.jpeg/.webp).
    Call after segmentation succeeds on the worker and before ``/complete`` when ``expose_masked_views`` is true.
    """
    if not remote_workers_enabled():
        raise HTTPException(
            status_code=503,
            detail="Remote workers disabled. Set APP_REMOTE_WORKERS=true on the API server.",
        )
    if job_manager.get_job(job_id) is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if not files:
        raise HTTPException(status_code=400, detail="Expected at least one file upload")

    dest = UPLOAD_DIR / job_id / "masked_views"
    dest.mkdir(parents=True, exist_ok=True)
    saved = 0
    for uf in files:
        safe = _safe_masked_view_basename(uf.filename or "")
        if safe is None:
            continue
        data = await uf.read()
        if len(data) < 32:
            continue
        (dest / safe).write_bytes(data)
        saved += 1

    if saved == 0:
        raise HTTPException(
            status_code=400,
            detail="No valid masked_* image parts (.png/.jpg/.jpeg/.webp)",
        )
    return {"job_id": job_id, "saved": saved, "path_prefix": f"uploads/{job_id}/masked_views/"}


@app.post("/internal/worker/jobs/{job_id}/fail", dependencies=[Depends(verify_worker_token)])
def internal_worker_fail(job_id: str, body: WorkerFailBody) -> JobRecord:
    try:
        return job_manager.fail_remote_job(job_id, body.error)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found") from None


@app.websocket("/internal/worker/ws")
async def internal_worker_websocket(websocket: WebSocket) -> None:
    """GPU workers subscribe for ``job_assigned`` messages when jobs enter ``QUEUED``.

    Connect with query param ``token`` equal to ``APP_WORKER_TOKEN``. Same TLS host as REST.
    """
    secret = os.getenv("APP_WORKER_TOKEN", "").strip()
    token = websocket.query_params.get("token")
    if not secret or token != secret:
        await websocket.close(code=4403)
        return
    if not remote_workers_enabled():
        await websocket.close(code=4503)
        return
    hub: WorkerHub = app.state.worker_hub
    await hub.register(websocket)
    try:
        # Keep the socket open for server → client ``job_assigned`` pushes only. Do not require text
        # from the worker; exiting cleanly on disconnect avoids a busy receive loop if the client
        # sends non-text frames (some proxies / libraries).
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    finally:
        hub.unregister(websocket)


def _register_worker_websocket_routes() -> None:
    """Also expose ``{APP_ROOT_PATH}/internal/worker/ws`` when behind a proxy that forwards the full path."""
    if not _root_path:
        return
    px = _root_path.rstrip("/")
    alt_path = f"{px}/internal/worker/ws".replace("//", "/")
    if not alt_path.startswith("/"):
        alt_path = "/" + alt_path
    app.router.add_websocket_route(alt_path, internal_worker_websocket)


_register_worker_websocket_routes()


@app.get(
    "/internal/worker/jobs/{job_id}/assignment",
    dependencies=[Depends(verify_worker_token)],
)
def internal_worker_assignment(job_id: str) -> dict[str, Any]:
    """Claim a specific ``QUEUED`` job and return settings + image URLs (same shape as ``/internal/worker/next``)."""
    if not remote_workers_enabled():
        raise HTTPException(
            status_code=503,
            detail="Remote workers disabled. Set APP_REMOTE_WORKERS=true on the API server.",
        )
    payload = job_manager.claim_remote_job_by_id(job_id)
    if payload is None:
        job = job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        raise HTTPException(
            status_code=409,
            detail=f"Job cannot be claimed (status={job.status.value})",
        )
    return payload


@app.post(
    "/internal/worker/jobs/{job_id}/progress",
    dependencies=[Depends(verify_worker_token)],
    response_model=JobRecord,
)
def internal_worker_progress(job_id: str, body: WorkerProgressBody) -> JobRecord:
    """Pipeline progress updates from the remote worker (typically every few seconds)."""
    if not remote_workers_enabled():
        raise HTTPException(
            status_code=503,
            detail="Remote workers disabled. Set APP_REMOTE_WORKERS=true on the API server.",
        )
    try:
        return job_manager.update_remote_worker_progress(job_id, stage=body.stage, progress=body.progress)
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found") from None
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@app.get("/job-stages")
def job_stages() -> dict[str, Any]:
    """Canonical JSON mapping of pipeline stage ids to progress bands (same file as ``ui/job-stages-progress.json``)."""
    if not _JOB_STAGES_JSON.is_file():
        raise HTTPException(status_code=404, detail="job-stages mapping file missing on server")
    return json.loads(_JOB_STAGES_JSON.read_text(encoding="utf-8"))


@app.get("/capture-guide")
def capture_guide() -> dict[str, Any]:
    """Capture UX + PUT /settings overlay hints for PolyCam-class quality (same file as ``ui/capture-guide.json``)."""
    if not _CAPTURE_GUIDE_JSON.is_file():
        raise HTTPException(status_code=404, detail="capture-guide file missing on server")
    return json.loads(_CAPTURE_GUIDE_JSON.read_text(encoding="utf-8"))


@app.get("/settings", response_model=RuntimeSettings)
def get_settings() -> RuntimeSettings:
    return settings_store.load()


@app.get("/settings/deployment")
def get_settings_deployment() -> dict[str, Any]:
    """Which ``PUT /settings`` fields matter on the API host vs the GPU worker, plus worker-only env vars."""
    return settings_deployment_guide()


@app.put("/settings", response_model=RuntimeSettings)
def update_settings(payload: RuntimeSettings) -> RuntimeSettings:
    return settings_store.save(payload)


@app.get("/server/status")
def server_status() -> dict:
    return collect_server_status(
        ROOT_DIR,
        settings=settings_store.load(),
        db_path=job_manager.db_path,
    )


@app.get("/jobs", response_model=list[JobRecord])
def list_jobs() -> list[JobRecord]:
    return [_decorate_job_response(j) for j in job_manager.list_jobs()]


@app.get("/jobs/{job_id}", response_model=JobRecord)
def get_job(job_id: str) -> JobRecord:
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return _decorate_job_response(job)


@app.get("/jobs/{job_id}/masked-preview", response_class=HTMLResponse, tags=["jobs"])
def job_masked_preview(job_id: str) -> HTMLResponse:
    """Browse SAM-isolated RGB frames (same files as ``masked_view_urls`` on ``GET /jobs/{job_id}``)."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    urls = _masked_view_urls(job_id)
    expose = settings_store.load().expose_masked_views
    jid_esc = html_module.escape(job_id)
    rows_parts: list[str] = []
    for u in urls:
        u_esc = html_module.escape(u, quote=True)
        rows_parts.append(
            f'<figure><img loading="lazy" src="{u_esc}" alt="masked"/>'
            f"<figcaption><a href=\"{u_esc}\">{html_module.escape(u)}</a></figcaption></figure>"
        )
    rows_html = "".join(rows_parts)
    hint = ""
    if not urls:
        hint = (
            "<p><strong>No masked views on disk yet.</strong> "
            "They appear after segmentation when <code>expose_masked_views</code> is true in "
            "<code>PUT /settings</code> (in-process pipeline mirrors when the job completes SAM; "
            "remote GPU workers POST files to <code>/internal/worker/jobs/{id}/masked-views</code>).</p>"
        )
        if not expose:
            hint += (
                "<p><strong>Tip:</strong> <code>expose_masked_views</code> is currently <strong>false</strong> "
                "— set it <strong>true</strong> and run again so copies are written under "
                "<code>uploads/{job_id}/masked_views/</code>.</p>"
            )
    body = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>Masked views — {jid_esc}</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 1rem 1.5rem; background: #111; color: #eee; }}
h1 {{ font-size: 1.15rem; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 1rem; }}
figure {{ margin: 0; background: #1a1a1a; padding: 0.5rem; border-radius: 8px; }}
img {{ width: 100%; height: auto; display: block; background: #000; border-radius: 4px; }}
figcaption {{ font-size: 0.72rem; word-break: break-all; margin-top: 0.35rem; }}
a {{ color: #8cf; }}
code {{ background: #222; padding: 0.12em 0.35em; border-radius: 4px; }}
</style></head><body>
<h1>SAM / abstraction preview — job <code>{jid_esc}</code></h1>
<p>{len(urls)} view(s). API fields: <code>masked_view_urls</code>, <code>masked_preview_page_url</code> on <code>GET /jobs/{jid_esc}</code>.</p>
{hint}
<div class="grid">{rows_html}</div>
</body></html>"""
    return HTMLResponse(content=body)


@app.post("/jobs/{job_id}/stop", response_model=JobRecord)
def stop_job(job_id: str) -> JobRecord:
    try:
        return _decorate_job_response(job_manager.request_stop(job_id))
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found") from None


@app.post("/jobs/{job_id}/continue", response_model=JobRecord)
def continue_job(job_id: str) -> JobRecord:
    try:
        return _decorate_job_response(job_manager.continue_job(job_id))
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found") from None
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@app.post("/jobs/{job_id}/reprocess", response_model=JobRecord)
def reprocess_job(job_id: str) -> JobRecord:
    try:
        return _decorate_job_response(job_manager.reprocess_job(job_id))
    except KeyError:
        raise HTTPException(status_code=404, detail="Job not found") from None
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@app.get("/models", response_model=list[ModelListItem])
def list_models() -> list[ModelListItem]:
    settings = settings_store.load()
    output_dir = ROOT_DIR / settings.output_dir_name
    base = _effective_model_base_url(settings)
    items = job_manager.list_models(output_dir, model_base_url=base)
    return [
        it.model_copy(update={"image_url": _image_sample_url(it.job_id)})
        for it in items
    ]


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
    return _decorate_job_response(job_manager.create_job_pending(use_job_id, len(files)))


@app.post("/jobs/{job_id}/start", response_model=JobRecord)
def start_job(job_id: str) -> JobRecord:
    """Begin processing for a PENDING job (requires at least 2 images on disk)."""
    try:
        return _decorate_job_response(job_manager.start_job(job_id))
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
        return _decorate_job_response(job_manager.enqueue_new_job(use_job_id, len(files)))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


def _safe_file_under(root: Path, rel: str) -> Path | None:
    """Resolve rel under root; reject path traversal; require a regular file."""
    root = root.resolve()
    rel_norm = rel.replace("\\", "/").strip("/")
    if not rel_norm:
        return None
    if ".." in Path(rel_norm).parts:
        return None
    candidate = (root / rel_norm).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


async def _serve_output_file(full_path: str) -> FileResponse:
    settings = settings_store.load()
    root = ROOT_DIR / settings.output_dir_name
    root.mkdir(parents=True, exist_ok=True)
    p = _safe_file_under(root, full_path)
    if not p:
        raise HTTPException(status_code=404, detail="Not found")
    mt, _ = mimetypes.guess_type(str(p))
    return FileResponse(
        str(p),
        filename=p.name,
        media_type=mt or "application/octet-stream",
    )


async def _serve_upload_file(full_path: str) -> FileResponse:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    p = _safe_file_under(UPLOAD_DIR, full_path)
    if not p:
        raise HTTPException(status_code=404, detail="Not found")
    mt, _ = mimetypes.guess_type(str(p))
    return FileResponse(
        str(p),
        filename=p.name,
        media_type=mt or "application/octet-stream",
    )


def _register_asset_file_routes() -> None:
    """Serve generated models and uploads via explicit GET/HEAD routes (not StaticFiles).

    Registers both ``/output`` / ``uploads`` and, when ``APP_ROOT_PATH`` is set,
    ``{APP_ROOT_PATH}/output`` / ``{APP_ROOT_PATH}/uploads`` so one Uvicorn process
    serves files correctly behind or without a reverse proxy.
    """
    prefixes: list[str] = [""]
    if _root_path:
        prefixes.append(_root_path.rstrip("/"))

    for px in prefixes:
        out_route = f"{px}/output/{{full_path:path}}" if px else "/output/{full_path:path}"
        up_route = f"{px}/uploads/{{full_path:path}}" if px else "/uploads/{full_path:path}"
        if not out_route.startswith("/"):
            out_route = "/" + out_route
        if not up_route.startswith("/"):
            up_route = "/" + up_route
        out_route = out_route.replace("//", "/")
        up_route = up_route.replace("//", "/")

        tag = (px.replace("/", "_") or "root").strip("_") or "root"
        app.add_api_route(
            out_route,
            _serve_output_file,
            methods=["GET", "HEAD"],
            name=f"files_output_{tag}",
            tags=["files"],
        )
        app.add_api_route(
            up_route,
            _serve_upload_file,
            methods=["GET", "HEAD"],
            name=f"files_uploads_{tag}",
            tags=["files"],
        )


_register_asset_file_routes()
