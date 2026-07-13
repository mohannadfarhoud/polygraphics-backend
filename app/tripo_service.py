"""Tripo image-to-3D jobs: persist state and run generation in API or GPU worker."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import uuid
from pathlib import Path
from typing import Callable

from . import tripo_db
from .tripo_core import resolve_tripo_backend, tripo_configured
from .tripo_models import TripoImageToModelResponse, TripoJobStatus, row_to_response

log = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class TripoService:
    def __init__(
        self,
        *,
        db_path: Path,
        root_dir: Path,
        download_url_for: Callable[[str], str],
    ) -> None:
        self.db_path = db_path
        self.root_dir = root_dir
        self._download_url_for = download_url_for
        tripo_db.init_schema(db_path)
        self._lock = threading.Lock()
        self._active: set[str] = set()

    def _upload_root(self) -> Path:
        rel = os.getenv("TRIPO_UPLOAD_DIR", "uploads/tripo").strip().strip("/")
        return self.root_dir / rel.replace("/", os.sep)

    def _max_bytes(self) -> int:
        mb = float(os.getenv("TRIPO_MAX_MB", "15"))
        return int(mb * 1024 * 1024)

    def max_upload_bytes(self) -> int:
        return self._max_bytes()

    def is_configured(self) -> bool:
        return tripo_configured()

    def create_job(
        self,
        *,
        image_bytes: bytes,
        filename: str,
        user_id: str = "",
    ) -> TripoImageToModelResponse:
        provider = resolve_tripo_backend()
        tripo_job_id = str(uuid.uuid4())
        base = self._upload_root() / tripo_job_id
        base.mkdir(parents=True, exist_ok=True)

        ext = Path(filename or "image.jpg").suffix.lower()
        if ext not in _IMAGE_EXTS:
            ext = ".jpg"
        input_path = base / f"input{ext}"
        input_path.write_bytes(image_bytes)
        model_path = base / "model.glb"

        row = tripo_db.insert_job(
            self.db_path,
            {
                "tripo_job_id": tripo_job_id,
                "user_id": user_id,
                "status": TripoJobStatus.queued.value,
                "input_image_path": str(input_path),
                "model_path": str(model_path),
                "provider": provider,
            },
        )
        if provider != "worker_triposr":
            self._schedule_worker(tripo_job_id=tripo_job_id, provider=provider)
        return self._to_response(row)

    def get_job(self, tripo_job_id: str) -> TripoImageToModelResponse | None:
        row = tripo_db.get_job(self.db_path, tripo_job_id)
        if not row:
            return None
        return self._to_response(row)

    def resolve_model_path(self, tripo_job_id: str) -> Path | None:
        row = tripo_db.get_job(self.db_path, tripo_job_id)
        if not row or row.get("status") != TripoJobStatus.completed.value:
            return None
        raw = row.get("model_path")
        if not raw:
            return None
        path = Path(str(raw))
        return path if path.is_file() else None

    def claim_next_for_worker(self) -> dict | None:
        row = tripo_db.claim_next_worker_job(self.db_path)
        if not row:
            return None
        image_path = Path(str(row["input_image_path"]))
        rel = image_path.relative_to(self.root_dir).as_posix()
        public = os.getenv("APP_PUBLIC_BASE_URL", "").strip().rstrip("/")
        image_url = f"{public}/{rel}" if public else f"/{rel}"
        return {
            "kind": "tripo_image_to_model",
            "tripo_job_id": row["tripo_job_id"],
            "provider": row.get("provider") or "worker_triposr",
            "image_url": image_url,
            "image_path": str(image_path),
        }

    def complete_from_worker(self, tripo_job_id: str, glb_bytes: bytes) -> TripoImageToModelResponse:
        row = tripo_db.get_job(self.db_path, tripo_job_id)
        if not row:
            raise KeyError(tripo_job_id)
        model_path = Path(str(row["model_path"]))
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(glb_bytes)
        if model_path.stat().st_size < 1024:
            raise ValueError("Uploaded GLB is too small")
        updated = tripo_db.update_job(
            self.db_path,
            tripo_job_id,
            status=TripoJobStatus.completed.value,
            tripo_task_id=f"worker-triposr-{tripo_job_id[:8]}",
            error=None,
        )
        assert updated is not None
        return self._to_response(updated)

    def fail_job(self, tripo_job_id: str, error: str) -> TripoImageToModelResponse:
        updated = tripo_db.update_job(
            self.db_path,
            tripo_job_id,
            status=TripoJobStatus.failed.value,
            error=error,
        )
        if updated is None:
            raise KeyError(tripo_job_id)
        return self._to_response(updated)

    def _to_response(self, row: dict) -> TripoImageToModelResponse:
        download_url = None
        if row.get("status") == TripoJobStatus.completed.value:
            model = row.get("model_path")
            if model and Path(str(model)).is_file():
                download_url = self._download_url_for(row["tripo_job_id"])
        return row_to_response(row, download_url=download_url)

    def _schedule_worker(self, *, tripo_job_id: str, provider: str) -> None:
        with self._lock:
            if tripo_job_id in self._active:
                return
            self._active.add(tripo_job_id)

        def _run() -> None:
            try:
                self._process_job(tripo_job_id=tripo_job_id, provider=provider)
            finally:
                with self._lock:
                    self._active.discard(tripo_job_id)

        threading.Thread(
            target=_run,
            name=f"tripo-{tripo_job_id[:8]}",
            daemon=True,
        ).start()

    def _process_job(self, *, tripo_job_id: str, provider: str) -> None:
        row = tripo_db.get_job(self.db_path, tripo_job_id)
        if not row:
            return
        tripo_db.update_job(
            self.db_path,
            tripo_job_id,
            status=TripoJobStatus.processing.value,
            error=None,
        )
        try:
            from .tripo_core import run_image_to_model

            input_path = Path(row["input_image_path"])
            model_path = Path(row["model_path"])
            task_id = asyncio.run(
                run_image_to_model(
                    image_path=input_path,
                    output_path=model_path,
                    backend=provider,
                )
            )
            tripo_db.update_job(
                self.db_path,
                tripo_job_id,
                status=TripoJobStatus.completed.value,
                tripo_task_id=task_id,
                error=None,
            )
        except Exception as exc:
            log.exception("tripo job %s failed", tripo_job_id)
            tripo_db.update_job(
                self.db_path,
                tripo_job_id,
                status=TripoJobStatus.failed.value,
                error=str(exc),
            )
