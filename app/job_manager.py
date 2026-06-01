from __future__ import annotations

import json
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .config import PipelineConfig
from .interfaces import JobRepository, JobStatus, WebSocketNotifier
from .job_models import JobRecord, ModelListItem
from . import jobs_db
from .pipeline import JobCancelled, ReconstructionPipeline
from .runtime_settings import RuntimeSettings, SettingsStore, minimum_input_images


def _mirror_masked_views_to_uploads(root_dir: Path, upload_dir: Path, masked_dir_name: str, job_id: str) -> None:
    """Expose ``masked/{job_id}/masked_*`` under ``uploads/{job_id}/masked_views/``."""
    src = root_dir / masked_dir_name / job_id
    if not src.is_dir():
        return
    dst = upload_dir / job_id / "masked_views"
    dst.mkdir(parents=True, exist_ok=True)
    for p in sorted(src.glob("masked_*")):
        if p.is_file():
            shutil.copy2(p, dst / p.name)


def _sorted_input_images(upload_dir: Path) -> list[Path]:
    if not upload_dir.is_dir():
        return []
    paths = sorted(upload_dir.glob("input_*"))
    return [p for p in paths if p.is_file()]


def remote_workers_enabled() -> bool:
    """When True, queued jobs are not executed in-process; a GPU worker polls ``/internal/worker/next``."""
    return os.getenv("APP_REMOTE_WORKERS", "").strip().lower() in ("1", "true", "yes")


def _relative_path_suffix(segment: str) -> str:
    """Same rule as ``APP_ROOT_PATH`` + segment for URL paths served by this app."""
    rp = os.getenv("APP_ROOT_PATH", "").strip()
    if rp and not rp.startswith("/"):
        rp = "/" + rp
    rp = rp.rstrip("/")
    suffix = "/" + segment.strip("/")
    return f"{rp}{suffix}" if rp else suffix


def _effective_model_base_url_for_jobs(settings: RuntimeSettings) -> str:
    for key in ("APP_MODEL_BASE_URL", "APP_CDN_BASE_URL"):
        v = os.getenv(key, "").strip()
        if v and "127.0.0.1" not in v and "localhost" not in v.lower():
            return v.rstrip("/")
    base = (settings.cdn_base_url or "").strip().rstrip("/")
    if base and "cdn.yoursite.com" not in base and "127.0.0.1" not in base:
        return base
    return _relative_path_suffix("output")


class PerJobJobRepository:
    def __init__(self, manager: JobManager, job_id: str) -> None:
        self._manager = manager
        self._job_id = job_id

    def set_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        stage: str | None = None,
        progress: int | None = None,
        model_url: str | None = None,
        model_format: str | None = None,
        error: str | None = None,
    ) -> None:
        self._manager.update_job(
            job_id,
            status,
            stage=stage,
            progress=progress,
            model_url=model_url,
            model_format=model_format,
            error=error,
        )


@dataclass
class JobManager:
    root_dir: Path
    settings_store: SettingsStore
    build_pipeline: Callable[[RuntimeSettings, JobRepository, WebSocketNotifier | None], ReconstructionPipeline]
    notifier: WebSocketNotifier | None = None

    def __post_init__(self) -> None:
        self._lock = threading.RLock()
        self._cancel_events: dict[str, threading.Event] = {}
        self._active_threads: dict[str, threading.Thread] = {}
        default_db = self.root_dir / "data" / "jobs.sqlite"
        raw = os.getenv("APP_DATABASE_PATH", "").strip()
        self._db_path = Path(raw).resolve() if raw else default_db.resolve()
        jobs_db.init_and_migrate(self._db_path, self.root_dir)
        self._recover_stuck_jobs()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def _recover_stuck_jobs(self) -> None:
        """Mark jobs left as PROCESSING/QUEUED by a previous process as STOPPED.

        The worker thread that was running them is gone (the API has restarted),
        so they can never finish on their own. The user can ``POST /jobs/{id}/continue``
        or ``/reprocess`` afterwards.
        """
        try:
            jobs = jobs_db.list_jobs(self._db_path)
        except Exception:
            return
        for j in jobs:
            if j.status in (JobStatus.PROCESSING, JobStatus.QUEUED):
                try:
                    self.update_job(
                        j.job_id,
                        JobStatus.STOPPED,
                        error="Stopped: API service restarted while this job was running",
                    )
                except Exception:
                    # never crash startup because of one malformed row
                    continue

    @property
    def upload_dir(self) -> Path:
        return self.root_dir / "uploads"

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return jobs_db.get_job(self._db_path, job_id)

    def list_jobs(self) -> list[JobRecord]:
        with self._lock:
            return jobs_db.list_jobs(self._db_path)

    def upsert_job(self, record: JobRecord) -> JobRecord:
        with self._lock:
            record.updated_at = time.time()
            jobs_db.save_record(self._db_path, record)
        return record

    def _delete_job_artifacts(self, job_id: str, settings: RuntimeSettings) -> None:
        """Remove files/directories associated with a job id."""
        output_dir = self.root_dir / settings.output_dir_name
        for ext in (".glb", ".ply"):
            p = output_dir / f"{job_id}{ext}"
            try:
                if p.is_file():
                    p.unlink()
            except OSError:
                pass
        for p in output_dir.glob(f"{job_id}_compare_*.glb"):
            try:
                if p.is_file():
                    p.unlink()
            except OSError:
                pass

        dir_candidates = [
            self.upload_dir / job_id,
            self.root_dir / settings.masked_dir_name / job_id,
            self.root_dir / settings.masks_dir_name / job_id,
            self.root_dir / "data" / "job_inputs" / job_id,
            self.root_dir / "data" / "ai_prior_workspace" / job_id,
            self.root_dir / "data" / "gs_workspace" / job_id,
            self.root_dir / "data" / "phase_scratch" / job_id,
            self.root_dir / "data" / "texture_abstraction" / job_id,
            self.root_dir / "data" / "surface_region_texture" / job_id,
        ]
        for d in dir_candidates:
            try:
                if d.is_dir():
                    shutil.rmtree(d, ignore_errors=True)
            except OSError:
                pass

    def delete_job(self, job_id: str) -> None:
        """Delete a job record and all known on-disk artifacts for this job id."""
        with self._lock:
            job = jobs_db.get_job(self._db_path, job_id)
            if not job:
                raise KeyError(job_id)
            if job.status in (JobStatus.QUEUED, JobStatus.PROCESSING):
                raise RuntimeError(
                    f"Cannot delete active job (status={job.status.value}). Stop it first, then delete."
                )
            active = self._active_threads.get(job_id)
            if active is not None and active.is_alive():
                raise RuntimeError("Cannot delete job while local worker thread is still running.")
            jobs_db.delete_job(self._db_path, job_id)
            self._cancel_events.pop(job_id, None)
            self._active_threads.pop(job_id, None)

        settings = self.settings_store.load()
        self._delete_job_artifacts(job_id, settings)

    def update_job(
        self,
        job_id: str,
        status: JobStatus,
        *,
        stage: str | None = None,
        progress: int | None = None,
        model_url: str | None = None,
        model_format: str | None = None,
        clear_model_url: bool = False,
        clear_error: bool = False,
        error: str | None = None,
    ) -> None:
        with self._lock:
            existing = jobs_db.get_job(self._db_path, job_id)
            if not existing:
                rec = JobRecord(
                    job_id=job_id,
                    status=status,
                    stage=stage,
                    progress=progress,
                    model_url=None if clear_model_url else model_url,
                    model_format=None if clear_model_url else model_format,
                    error=None if clear_error else error,
                )
            else:
                # Don't let a still-running worker resurrect a job the user has stopped.
                # Transitioning from STOPPED back to PROCESSING/QUEUED is only allowed via
                # continue_job/reprocess_job (those go through QUEUED with cleared events).
                # Block STOPPED → PROCESSING only (orphan guard). Allow STOPPED → QUEUED
                # so ``reprocess`` / ``continue`` can re-queue after a user stop.
                if existing.status == JobStatus.STOPPED and status == JobStatus.PROCESSING:
                    return
                rec = existing
                rec.status = status
                if stage is not None:
                    rec.stage = stage
                if progress is not None:
                    rec.progress = max(0, min(100, int(progress)))
                if status == JobStatus.COMPLETED:
                    rec.progress = 100
                elif status in (JobStatus.QUEUED, JobStatus.PENDING):
                    rec.progress = 0
                if clear_model_url:
                    rec.model_url = None
                    rec.model_format = None
                else:
                    if model_url is not None:
                        rec.model_url = model_url
                    if model_format is not None:
                        rec.model_format = model_format
                if clear_error:
                    rec.error = None
                elif error is not None:
                    rec.error = error
                rec.updated_at = time.time()
            jobs_db.save_record(self._db_path, rec)
            notify_url = rec.model_url
            notify_fmt = rec.model_format
            notify_err = rec.error
            notify_stage = rec.stage
            notify_progress = rec.progress
        if self.notifier:
            self.notifier.notify_job_update(
                job_id,
                status,
                stage=notify_stage,
                progress=notify_progress,
                model_url=notify_url,
                model_format=notify_fmt,
                error=notify_err,
            )
        if status == JobStatus.QUEUED and remote_workers_enabled():
            try:
                from .worker_hub import schedule_worker_job_notice

                schedule_worker_job_notice(job_id)
            except Exception:
                pass

    def create_job_pending(self, job_id: str, image_count: int) -> JobRecord:
        """Register uploads only; processing starts after start_job()."""
        rec = JobRecord(job_id=job_id, status=JobStatus.PENDING, image_count=image_count)
        self.upsert_job(rec)
        return self.get_job(job_id) or rec

    def start_job(self, job_id: str) -> JobRecord:
        """Move PENDING → QUEUED and run the worker."""
        job = self.get_job(job_id)
        if not job:
            raise KeyError(job_id)
        if job.status != JobStatus.PENDING:
            raise RuntimeError(f"Can only start a PENDING job; current status is {job.status.value}")
        settings = self.settings_store.load()
        min_images = int(minimum_input_images(settings))
        paths = _sorted_input_images(self.upload_dir / job_id)
        if len(paths) < min_images:
            raise RuntimeError(
                f"Need at least {min_images} image(s) under uploads/{{job_id}}/ before starting"
            )
        self._cancel_events[job_id] = threading.Event()
        self.update_job(job_id, JobStatus.QUEUED, clear_model_url=True, clear_error=True)
        if not remote_workers_enabled():
            self._start_worker(job_id)
        return self.get_job(job_id) or job

    def enqueue_new_job(self, job_id: str, image_count: int) -> JobRecord:
        """Create PENDING record and immediately start (same as upload + start)."""
        self.create_job_pending(job_id, image_count)
        return self.start_job(job_id)

    def request_stop(self, job_id: str) -> JobRecord:
        """Stop a job in any non-terminal state.

        Three cases handled:
          * **PENDING / QUEUED**: never started, so we just write STOPPED.
          * **PROCESSING + live worker thread**: signal the cancel event so the worker
            exits cleanly between stages, **and** write STOPPED right away so the UI
            reflects the user's intent without waiting for the next stage boundary.
          * **PROCESSING + no live worker** (orphan after a service restart):
            no thread will ever read the cancel event; force the DB to STOPPED.
        Already-terminal jobs (COMPLETED/FAILED/STOPPED) are returned unchanged.
        """
        job = self.get_job(job_id)
        if not job:
            raise KeyError(job_id)

        ev = self._cancel_events.setdefault(job_id, threading.Event())
        ev.set()

        with self._lock:
            thread = self._active_threads.get(job_id)
            thread_alive = thread is not None and thread.is_alive()

        if job.status in (JobStatus.PENDING, JobStatus.QUEUED):
            self.update_job(
                job_id,
                JobStatus.STOPPED,
                error="Cancelled before processing started",
            )
        elif job.status == JobStatus.PROCESSING:
            if thread_alive:
                self.update_job(job_id, JobStatus.STOPPED, error="Stopped by user")
            else:
                self.update_job(
                    job_id,
                    JobStatus.STOPPED,
                    error="Stopped (orphaned: worker thread is no longer running, likely a service restart)",
                )
        return self.get_job(job_id) or job

    def continue_job(self, job_id: str) -> JobRecord:
        job = self.get_job(job_id)
        if not job:
            raise KeyError(job_id)
        if job.status == JobStatus.PROCESSING:
            raise RuntimeError("Job is already processing")
        if job.status == JobStatus.QUEUED:
            if remote_workers_enabled():
                try:
                    from .worker_hub import schedule_worker_job_notice

                    schedule_worker_job_notice(job_id)
                except Exception:
                    pass
                return self.get_job(job_id) or job
            raise RuntimeError("Job is already queued")
        if job.status == JobStatus.PENDING:
            raise RuntimeError("Job has not started yet; use POST /jobs/{job_id}/start")
        if job.status not in (JobStatus.STOPPED, JobStatus.FAILED, JobStatus.PAUSED):
            raise RuntimeError(
                f"Use reprocess for completed jobs. Cannot continue from status {job.status.value}"
            )
        settings = self.settings_store.load()
        min_images = int(minimum_input_images(settings))
        paths = _sorted_input_images(self.upload_dir / job_id)
        if len(paths) < min_images:
            raise RuntimeError(
                f"Not enough input images to continue; need at least {min_images} image(s) under uploads/{{job_id}}/"
            )
        self._cancel_events[job_id] = threading.Event()
        self.update_job(job_id, JobStatus.QUEUED, clear_model_url=True, clear_error=True)
        if not remote_workers_enabled():
            self._start_worker(job_id)
        return self.get_job(job_id) or job

    def reprocess_job(self, job_id: str) -> JobRecord:
        job = self.get_job(job_id)
        if not job:
            raise KeyError(job_id)
        if job.status == JobStatus.PROCESSING:
            raise RuntimeError("Job is already processing")
        if job.status == JobStatus.QUEUED:
            if remote_workers_enabled():
                try:
                    from .worker_hub import schedule_worker_job_notice

                    schedule_worker_job_notice(job_id)
                except Exception:
                    pass
                return self.get_job(job_id) or job
            raise RuntimeError("Job is already queued")
        if job.status == JobStatus.PENDING:
            raise RuntimeError("Job has not started yet; use POST /jobs/{job_id}/start")
        settings = self.settings_store.load()
        min_images = int(minimum_input_images(settings))
        paths = _sorted_input_images(self.upload_dir / job_id)
        if len(paths) < min_images:
            raise RuntimeError(
                f"Not enough input images; need at least {min_images} image(s) under uploads/{{job_id}}/"
            )
        self._cancel_events[job_id] = threading.Event()
        self.update_job(job_id, JobStatus.QUEUED, clear_model_url=True, clear_error=True)
        if not remote_workers_enabled():
            self._start_worker(job_id)
        return self.get_job(job_id) or job

    def list_models(self, output_dir: Path, *, model_base_url: str) -> list[ModelListItem]:
        if not output_dir.is_dir():
            return []
        files: list[Path] = []
        for ext in ("*.glb", "*.ply"):
            files.extend(output_dir.glob(ext))
        items: list[ModelListItem] = []
        base = model_base_url.rstrip("/")
        for p in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True):
            st = p.stat()
            jid = p.stem
            items.append(
                ModelListItem(
                    job_id=jid,
                    filename=p.name,
                    url=f"{base}/{p.name}",
                    size_bytes=st.st_size,
                    modified_at=st.st_mtime,
                )
            )
        return items

    def _build_assignment_payload(self, job_id: str) -> dict:
        settings = self.settings_store.load()
        paths = _sorted_input_images(self.upload_dir / job_id)
        public = os.getenv("APP_PUBLIC_BASE_URL", "").strip().rstrip("/")
        image_urls: list[str] = []
        for p in paths:
            if public:
                image_urls.append(f"{public}/uploads/{job_id}/{p.name}")
            else:
                image_urls.append(f"/uploads/{job_id}/{p.name}")
        job = self.get_job(job_id)
        return {
            "job_id": job_id,
            "image_count": job.image_count if job else 0,
            "settings": json.loads(settings.model_dump_json()),
            "image_urls": image_urls,
        }

    def claim_remote_job_by_id(self, job_id: str) -> dict | None:
        """Atomically claim a specific ``QUEUED`` job (``QUEUED`` → ``PROCESSING``)."""
        with self._lock:
            job = jobs_db.get_job(self._db_path, job_id)
            if not job or job.status != JobStatus.QUEUED:
                return None
            self._cancel_events[job_id] = threading.Event()
            self.update_job(
                job_id,
                JobStatus.PROCESSING,
                stage="starting",
                progress=5,
                clear_error=True,
            )
        return self._build_assignment_payload(job_id)

    def update_remote_worker_progress(self, job_id: str, *, stage: str, progress: int) -> JobRecord:
        job = self.get_job(job_id)
        if not job:
            raise KeyError(job_id)
        if job.status != JobStatus.PROCESSING:
            raise RuntimeError(f"Job {job_id} is not PROCESSING (got {job.status.value})")
        self.update_job(job_id, JobStatus.PROCESSING, stage=stage, progress=progress)
        result = self.get_job(job_id)
        if not result:
            raise RuntimeError("Job disappeared")
        return result

    def claim_next_remote_job(self) -> dict | None:
        """Atomically pick the oldest ``QUEUED`` job and move it to ``PROCESSING``."""
        with self._lock:
            queued = jobs_db.list_queued_oldest_first(self._db_path)
            if not queued:
                return None
            job = queued[0]
            jid = job.job_id
            self._cancel_events[jid] = threading.Event()
            self.update_job(
                jid,
                JobStatus.PROCESSING,
                stage="starting",
                progress=5,
                clear_error=True,
            )
        return self._build_assignment_payload(jid)

    def complete_remote_job(self, job_id: str, file_data: bytes, model_format: str) -> JobRecord:
        job = self.get_job(job_id)
        if not job:
            raise KeyError(job_id)
        if job.status != JobStatus.PROCESSING:
            raise RuntimeError(f"Job {job_id} is not PROCESSING (got {job.status.value})")
        ext = model_format.lower().lstrip(".")
        if ext not in ("glb", "ply"):
            raise ValueError("model_format must be glb or ply")
        settings = self.settings_store.load()
        out_dir = self.root_dir / settings.output_dir_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{job_id}.{ext}"
        out_path.write_bytes(file_data)
        base = _effective_model_base_url_for_jobs(settings)
        if base.startswith("/"):
            model_url = f"{base.rstrip('/')}/{job_id}.{ext}"
        else:
            model_url = f"{base.rstrip('/')}/{job_id}.{ext}"
        self.update_job(
            job_id,
            JobStatus.COMPLETED,
            stage="completed",
            progress=100,
            model_url=model_url,
            model_format=ext,
            clear_error=True,
        )
        result = self.get_job(job_id)
        if not result:
            raise RuntimeError("Job disappeared after complete")
        return result

    def save_remote_comparison_glb(self, job_id: str, variant: str, file_data: bytes) -> dict[str, str]:
        """Write ``{job_id}_compare_{variant}.glb`` while the job is still PROCESSING (before primary ``/complete``)."""
        job = self.get_job(job_id)
        if not job:
            raise KeyError(job_id)
        if job.status != JobStatus.PROCESSING:
            raise RuntimeError(
                f"Job {job_id} is not PROCESSING (got {job.status.value}); comparison GLB upload is only valid mid-run."
            )
        v = variant.strip().lower()
        if v not in ("mesh",):
            raise ValueError("variant must be mesh")
        settings = self.settings_store.load()
        out_dir = self.root_dir / settings.output_dir_name
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{job_id}_compare_{v}.glb"
        out_path = out_dir / fname
        out_path.write_bytes(file_data)
        base = _effective_model_base_url_for_jobs(settings)
        model_url = f"{base.rstrip('/')}/{fname}"
        return {"filename": fname, "model_url": model_url}

    def fail_remote_job(self, job_id: str, error: str) -> JobRecord:
        self.update_job(job_id, JobStatus.FAILED, error=error)
        result = self.get_job(job_id)
        if not result:
            raise KeyError(job_id)
        return result

    def _run_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if not job:
            return
        if job.status != JobStatus.QUEUED:
            return
        settings = self.settings_store.load()
        min_images = int(minimum_input_images(settings))
        paths = _sorted_input_images(self.upload_dir / job_id)
        if len(paths) < min_images:
            self.update_job(
                job_id,
                JobStatus.FAILED,
                error=f"Not enough input images (need at least {min_images})",
            )
            return
        cancel_ev = self._cancel_events.setdefault(job_id, threading.Event())
        job_repo = PerJobJobRepository(self, job_id)
        pipe = self.build_pipeline(settings, job_repo, None)
        try:
            try:
                from app.cuda_memory import bootstrap_worker_cuda

                bootstrap_worker_cuda()
            except Exception:
                pass
            pipe.process_3d_job(job_id, paths, cancel_event=cancel_ev)
            if settings.expose_masked_views:
                _mirror_masked_views_to_uploads(
                    self.root_dir,
                    self.upload_dir,
                    settings.masked_dir_name,
                    job_id,
                )
        except JobCancelled:
            self.update_job(job_id, JobStatus.STOPPED, error="Stopped by user")
        except Exception as exc:
            self.update_job(job_id, JobStatus.FAILED, error=str(exc))

    def _start_worker(self, job_id: str) -> None:
        def run() -> None:
            try:
                self._run_job(job_id)
            finally:
                with self._lock:
                    self._active_threads.pop(job_id, None)

        with self._lock:
            existing = self._active_threads.get(job_id)
            if existing is not None and existing.is_alive():
                return
            t = threading.Thread(target=run, name=f"job-{job_id}", daemon=True)
            self._active_threads[job_id] = t
            t.start()
