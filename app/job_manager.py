from __future__ import annotations

import os
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
from .runtime_settings import RuntimeSettings, SettingsStore


def _sorted_input_images(upload_dir: Path) -> list[Path]:
    if not upload_dir.is_dir():
        return []
    paths = sorted(upload_dir.glob("input_*"))
    return [p for p in paths if p.is_file()]


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
        self._lock = threading.Lock()
        self._cancel_events: dict[str, threading.Event] = {}
        self._active_threads: dict[str, threading.Thread] = {}
        default_db = self.root_dir / "data" / "jobs.sqlite"
        raw = os.getenv("APP_DATABASE_PATH", "").strip()
        self._db_path = Path(raw).resolve() if raw else default_db.resolve()
        jobs_db.init_and_migrate(self._db_path, self.root_dir)

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
        paths = _sorted_input_images(self.upload_dir / job_id)
        if len(paths) < 2:
            raise RuntimeError("Need at least 2 images under uploads/{job_id}/ before starting")
        self._cancel_events[job_id] = threading.Event()
        self.update_job(job_id, JobStatus.QUEUED, clear_model_url=True, clear_error=True)
        self._start_worker(job_id)
        return self.get_job(job_id) or job

    def enqueue_new_job(self, job_id: str, image_count: int) -> JobRecord:
        """Create PENDING record and immediately start (same as upload + start)."""
        self.create_job_pending(job_id, image_count)
        return self.start_job(job_id)

    def request_stop(self, job_id: str) -> JobRecord:
        job = self.get_job(job_id)
        if not job:
            raise KeyError(job_id)
        ev = self._cancel_events.setdefault(job_id, threading.Event())
        ev.set()
        if job.status == JobStatus.PENDING:
            self.update_job(job_id, JobStatus.STOPPED, error="Cancelled before processing started")
        elif job.status == JobStatus.QUEUED:
            self.update_job(job_id, JobStatus.STOPPED, error="Cancelled before processing started")
        return self.get_job(job_id) or job

    def continue_job(self, job_id: str) -> JobRecord:
        job = self.get_job(job_id)
        if not job:
            raise KeyError(job_id)
        if job.status == JobStatus.PROCESSING:
            raise RuntimeError("Job is already processing")
        if job.status == JobStatus.QUEUED:
            raise RuntimeError("Job is already queued")
        if job.status == JobStatus.PENDING:
            raise RuntimeError("Job has not started yet; use POST /jobs/{job_id}/start")
        if job.status not in (JobStatus.STOPPED, JobStatus.FAILED, JobStatus.PAUSED):
            raise RuntimeError(
                f"Use reprocess for completed jobs. Cannot continue from status {job.status.value}"
            )
        paths = _sorted_input_images(self.upload_dir / job_id)
        if len(paths) < 2:
            raise RuntimeError("Not enough input images to continue; need at least 2 images under uploads/{job_id}/")
        self._cancel_events[job_id] = threading.Event()
        self.update_job(job_id, JobStatus.QUEUED, clear_model_url=True, clear_error=True)
        self._start_worker(job_id)
        return self.get_job(job_id) or job

    def reprocess_job(self, job_id: str) -> JobRecord:
        job = self.get_job(job_id)
        if not job:
            raise KeyError(job_id)
        if job.status == JobStatus.PROCESSING:
            raise RuntimeError("Job is already processing")
        if job.status == JobStatus.QUEUED:
            raise RuntimeError("Job is already queued")
        if job.status == JobStatus.PENDING:
            raise RuntimeError("Job has not started yet; use POST /jobs/{job_id}/start")
        paths = _sorted_input_images(self.upload_dir / job_id)
        if len(paths) < 2:
            raise RuntimeError("Not enough input images; need at least 2 images under uploads/{job_id}/")
        self._cancel_events[job_id] = threading.Event()
        self.update_job(job_id, JobStatus.QUEUED, clear_model_url=True, clear_error=True)
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
            items.append(
                ModelListItem(
                    job_id=p.stem,
                    filename=p.name,
                    url=f"{base}/{p.name}",
                    size_bytes=st.st_size,
                    modified_at=st.st_mtime,
                )
            )
        return items

    def _run_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        if not job:
            return
        if job.status != JobStatus.QUEUED:
            return
        paths = _sorted_input_images(self.upload_dir / job_id)
        if len(paths) < 2:
            self.update_job(job_id, JobStatus.FAILED, error="Not enough input images (need at least 2)")
            return
        settings = self.settings_store.load()
        cancel_ev = self._cancel_events.setdefault(job_id, threading.Event())
        job_repo = PerJobJobRepository(self, job_id)
        pipe = self.build_pipeline(settings, job_repo, None)
        try:
            pipe.process_3d_job(job_id, paths, cancel_event=cancel_ev)
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
