"""Photo try-on compose: validation, asset resolution, async worker."""

from __future__ import annotations

import logging
import os
import shutil
import threading
import uuid
from pathlib import Path
from urllib.parse import urlparse

import httpx

from . import photo_compose_db
from .job_manager import JobManager
from .photo_compose_models import PhotoComposeResponse, PhotoComposeStatus, row_to_response
from .photo_compose_util import placement_pixels
from .runtime_settings import SettingsStore

log = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_PLACEMENT_SIDES = frozenset({"left", "right", "auto"})


class PhotoComposeService:
    def __init__(
        self,
        *,
        db_path: Path,
        root_dir: Path,
        job_manager: JobManager,
        settings_store: SettingsStore,
        uploads_dir: Path,
    ) -> None:
        self.db_path = db_path
        self.root_dir = root_dir
        self.job_manager = job_manager
        self.settings_store = settings_store
        self.uploads_dir = uploads_dir
        photo_compose_db.init_schema(db_path)
        self._lock = threading.Lock()
        self._active: set[str] = set()

    def _compose_upload_root(self) -> Path:
        rel = os.getenv("PHOTO_COMPOSE_UPLOAD_DIR", "uploads/photo-compose").strip().strip("/")
        return self.root_dir / rel.replace("/", os.sep)

    def _max_bytes(self) -> int:
        mb = float(os.getenv("PHOTO_COMPOSE_MAX_MB", "10"))
        return int(mb * 1024 * 1024)

    def max_upload_bytes(self) -> int:
        return self._max_bytes()

    def _model_output_exists(self, job_id: str) -> bool:
        settings = self.settings_store.load()
        out = self.root_dir / settings.output_dir_name
        if not out.is_dir():
            return False
        return (out / f"{job_id}.glb").is_file() or (out / f"{job_id}.ply").is_file()

    def assert_model_exists(self, job_id: str) -> None:
        job = self.job_manager.get_job(job_id)
        if job is None and not self._model_output_exists(job_id):
            raise KeyError(job_id)

    def resolve_model_image_url(self, job_id: str) -> str:
        """Same logic as GET /models image_url."""
        job_dir = self.uploads_dir / job_id
        if job_dir.is_dir():
            for p in sorted(job_dir.glob("input_*")):
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
                    return f"/uploads/{job_id}/{p.name}"
        raise ValueError("Model has no product image (image_url). Upload a thumbnail for this job.")

    def _url_to_local_path(self, url: str) -> Path | None:
        u = url.strip()
        if not u:
            return None
        if u.startswith("/"):
            # strip optional APP_ROOT_PATH prefix
            rp = os.getenv("APP_ROOT_PATH", "").strip().rstrip("/")
            if rp and u.startswith(rp + "/"):
                u = u[len(rp) :]
            rel = u.lstrip("/")
            if rel.startswith("uploads/"):
                return self.root_dir / rel.replace("/", os.sep)
            return self.root_dir / rel.replace("/", os.sep)
        parsed = urlparse(u)
        if parsed.scheme in ("http", "https") and parsed.path:
            path = parsed.path
            rp = os.getenv("APP_ROOT_PATH", "").strip().rstrip("/")
            if rp and path.startswith(rp + "/"):
                path = path[len(rp) :]
            if "/uploads/" in path:
                idx = path.index("/uploads/")
                rel = path[idx + 1 :]
                return self.root_dir / rel.replace("/", os.sep)
        return None

    def fetch_earring_image(self, image_url: str, dest: Path) -> Path:
        local = self._url_to_local_path(image_url)
        if local is not None and local.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(local.read_bytes())
            return dest
        if image_url.startswith("http://") or image_url.startswith("https://"):
            dest.parent.mkdir(parents=True, exist_ok=True)
            with httpx.Client(timeout=60.0) as client:
                r = client.get(image_url)
                r.raise_for_status()
                dest.write_bytes(r.content)
            return dest
        raise FileNotFoundError(f"Cannot resolve earring image_url: {image_url}")

    def create_compose_job(
        self,
        *,
        job_id: str,
        face_bytes: bytes,
        face_filename: str,
        placement_x: float,
        placement_y: float,
        image_width: int,
        image_height: int,
        placement_side: str,
        user_prompt: str | None,
        user_id: str | None,
    ) -> PhotoComposeResponse:
        self.assert_model_exists(job_id)
        image_url = self.resolve_model_image_url(job_id)

        compose_id = str(uuid.uuid4())
        base = self._compose_upload_root() / compose_id
        base.mkdir(parents=True, exist_ok=True)

        ext = Path(face_filename or "face.jpg").suffix.lower()
        if ext not in (".jpg", ".jpeg", ".png"):
            ext = ".jpg"
        face_path = base / f"face{ext}"
        face_path.write_bytes(face_bytes)

        result_rel = f"/uploads/photo-compose/{compose_id}/result.jpg"
        row = photo_compose_db.insert_job(
            self.db_path,
            {
                "compose_id": compose_id,
                "job_id": job_id,
                "user_id": user_id,
                "status": PhotoComposeStatus.queued.value,
                "placement_x": placement_x,
                "placement_y": placement_y,
                "image_width": image_width,
                "image_height": image_height,
                "placement_side": placement_side,
                "face_image_path": str(face_path),
                "user_prompt": user_prompt,
            },
        )

        self._schedule_worker(
            compose_id=compose_id,
            image_url=image_url,
            result_rel=result_rel,
        )
        return row_to_response(row)

    def get_compose(self, compose_id: str) -> PhotoComposeResponse | None:
        row = photo_compose_db.get_job(self.db_path, compose_id)
        if not row:
            return None
        return row_to_response(row)

    def _schedule_worker(self, *, compose_id: str, image_url: str, result_rel: str) -> None:
        with self._lock:
            if compose_id in self._active:
                return
            self._active.add(compose_id)

        def _run() -> None:
            try:
                self._process_compose(compose_id=compose_id, image_url=image_url, result_rel=result_rel)
            finally:
                with self._lock:
                    self._active.discard(compose_id)

        threading.Thread(target=_run, name=f"photo-compose-{compose_id[:8]}", daemon=True).start()

    def _process_compose(self, *, compose_id: str, image_url: str, result_rel: str) -> None:
        row = photo_compose_db.get_job(self.db_path, compose_id)
        if not row:
            return
        photo_compose_db.update_job(
            self.db_path,
            compose_id,
            status=PhotoComposeStatus.processing.value,
            error=None,
        )
        try:
            from .photo_compose_runner import compose_photorealistic, draw_placement_marker

            face_path = Path(row["face_image_path"])
            px, py = placement_pixels(
                float(row["placement_x"]),
                float(row["placement_y"]),
                int(row["image_width"]),
                int(row["image_height"]),
            )
            base = face_path.parent
            marked_path = base / "face_marked.jpg"
            dev_mock = os.getenv("PHOTO_COMPOSE_DEV_MOCK", "").strip().lower() in ("1", "true", "yes")
            gemini_ok = bool(os.getenv("GEMINI_API_KEY", "").strip())
            if dev_mock and not gemini_ok:
                shutil.copy2(face_path, marked_path)
            else:
                draw_placement_marker(face_path, px=px, py=py, out_path=marked_path)

            earring_path = base / "earring.jpg"
            self.fetch_earring_image(image_url, earring_path)

            result_path = base / "result.jpg"
            revised = compose_photorealistic(
                marked_face_path=marked_path,
                earring_path=earring_path,
                result_path=result_path,
                px=px,
                py=py,
                image_width=int(row["image_width"]),
                image_height=int(row["image_height"]),
                placement_side=str(row.get("placement_side") or "auto"),
                user_prompt=row.get("user_prompt"),
            )
            photo_compose_db.update_job(
                self.db_path,
                compose_id,
                status=PhotoComposeStatus.completed.value,
                result_url=result_rel,
                revised_prompt=revised,
                error=None,
            )
        except Exception as exc:
            log.exception("photo compose %s failed", compose_id)
            photo_compose_db.update_job(
                self.db_path,
                compose_id,
                status=PhotoComposeStatus.failed.value,
                error=str(exc),
            )
