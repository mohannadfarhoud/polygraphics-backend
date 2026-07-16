"""Isolation dataset, training, model registry, and predict orchestration."""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import uuid
import zipfile
from pathlib import Path
from typing import Any, Callable

from . import isolation_db
from .isolation_inference import invalidate_session, predict_isolated_png
from .isolation_mask import generate_and_save_mask
from .isolation_models import (
    IsolationDatasetDetail,
    IsolationDatasetSummary,
    IsolationHealthResponse,
    IsolationModelSummary,
    IsolationStatistics,
    IsolationTrainMetrics,
    IsolationTrainResponse,
    IsolationTrainStatus,
    row_to_dataset_summary,
)
from .isolation_quota import (
    IsolationQuotaExceededError,
    charge_isolation_predict,
    get_isolation_quota,
    init_schema as init_quota_schema,
)

log = logging.getLogger(__name__)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_MIN_PAIRS_WARN = 20


class IsolationService:
    def __init__(
        self,
        *,
        db_path: Path,
        root_dir: Path,
        asset_url_for: Callable[[str], str],
    ) -> None:
        self.db_path = db_path
        self.root_dir = root_dir
        self._asset_url_for = asset_url_for
        isolation_db.init_schema(db_path)
        init_quota_schema(db_path)
        self._lock = threading.Lock()
        self._active_train: set[str] = set()

    def _datasets_root(self) -> Path:
        rel = os.getenv("ISOLATION_DATASETS_DIR", "datasets/isolation").strip().strip("/")
        return self.root_dir / rel.replace("/", os.sep)

    def _models_root(self) -> Path:
        rel = os.getenv("ISOLATION_MODELS_DIR", "models/isolation").strip().strip("/")
        return self.root_dir / rel.replace("/", os.sep)

    def _predict_root(self) -> Path:
        rel = os.getenv("ISOLATION_PREDICT_DIR", "uploads/isolation/predict").strip().strip("/")
        return self.root_dir / rel.replace("/", os.sep)

    def _dataset_dir(self, dataset_id: str) -> Path:
        return self._datasets_root() / dataset_id

    def _model_dir(self, model_id: str) -> Path:
        return self._models_root() / model_id

    def _meta_path(self, dataset_id: str) -> Path:
        return self._dataset_dir(dataset_id) / "meta.json"

    def _load_meta(self, dataset_id: str) -> dict[str, Any]:
        path = self._meta_path(dataset_id)
        if not path.is_file():
            return {"pairs": []}
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_meta(self, dataset_id: str, meta: dict[str, Any]) -> None:
        path = self._meta_path(dataset_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def create_dataset(self, *, name: str, owner_user_id: str | None) -> IsolationDatasetSummary:
        row = isolation_db.insert_dataset(self.db_path, name=name, owner_user_id=owner_user_id)
        ddir = self._dataset_dir(row["dataset_id"])
        ddir.mkdir(parents=True, exist_ok=True)
        self._save_meta(row["dataset_id"], {"pairs": [], "name": name})
        return row_to_dataset_summary(row)

    def list_datasets(self) -> list[IsolationDatasetSummary]:
        return [row_to_dataset_summary(r) for r in isolation_db.list_datasets(self.db_path)]

    def get_dataset(self, dataset_id: str) -> IsolationDatasetDetail | None:
        row = isolation_db.get_dataset(self.db_path, dataset_id)
        if not row:
            return None
        meta = self._load_meta(dataset_id)
        summary = row_to_dataset_summary(row)
        return IsolationDatasetDetail(**summary.model_dump(), pairs=list(meta.get("pairs") or []))

    def delete_dataset(self, dataset_id: str) -> bool:
        ok = isolation_db.delete_dataset(self.db_path, dataset_id)
        if ok:
            shutil.rmtree(self._dataset_dir(dataset_id), ignore_errors=True)
        return ok

    def _ext_from_filename(self, filename: str) -> str:
        ext = Path(filename or "image.jpg").suffix.lower()
        return ext if ext in _IMAGE_EXTS else ".jpg"

    def add_pairs(
        self,
        *,
        dataset_id: str,
        items: list[dict[str, Any]],
    ) -> tuple[int, list[int]]:
        row = isolation_db.get_dataset(self.db_path, dataset_id)
        if not row:
            raise KeyError(dataset_id)
        meta = self._load_meta(dataset_id)
        pairs_meta: list[dict[str, Any]] = list(meta.get("pairs") or [])
        used_indices = {int(p["index"]) for p in pairs_meta}
        uploaded = 0
        indices: list[int] = []

        for item in items:
            raw_index = item.get("index")
            if raw_index is None:
                index = 0
                while index in used_indices:
                    index += 1
            else:
                index = int(raw_index)
            before_bytes: bytes = item["before_bytes"]
            after_bytes: bytes = item["after_bytes"]
            mask_bytes: bytes | None = item.get("mask_bytes")
            before_name = item.get("before_name") or "before.jpg"
            after_name = item.get("after_name") or "after.jpg"

            pair_dir = self._dataset_dir(dataset_id) / "pairs" / str(index)
            pair_dir.mkdir(parents=True, exist_ok=True)

            before_ext = self._ext_from_filename(before_name)
            after_ext = self._ext_from_filename(after_name)
            before_path = pair_dir / f"before{before_ext}"
            after_path = pair_dir / f"after{after_ext}"
            mask_path = pair_dir / "mask.png"

            before_path.write_bytes(before_bytes)
            after_path.write_bytes(after_bytes)
            if mask_bytes:
                mask_path.write_bytes(mask_bytes)
            else:
                generate_and_save_mask(
                    before_path=str(before_path),
                    after_path=str(after_path),
                    mask_path=str(mask_path),
                )

            entry = {
                "index": index,
                "before": f"pairs/{index}/before{before_ext}",
                "after": f"pairs/{index}/after{after_ext}",
                "mask": f"pairs/{index}/mask.png",
            }
            pairs_meta = [p for p in pairs_meta if int(p["index"]) != index]
            pairs_meta.append(entry)
            used_indices.add(index)
            uploaded += 1
            indices.append(index)

        pairs_meta.sort(key=lambda p: int(p["index"]))
        meta["pairs"] = pairs_meta
        self._save_meta(dataset_id, meta)
        isolation_db.update_dataset(self.db_path, dataset_id, pair_count=len(pairs_meta))
        return uploaded, sorted(indices)

    def record_upload_event(
        self,
        *,
        dataset_id: str,
        user_id: str | None,
        submitted_photos: int,
        submitted_pairs: int,
        successful_pairs: int,
        error: str | None = None,
    ) -> None:
        try:
            isolation_db.insert_upload_event(
                self.db_path,
                dataset_id=dataset_id,
                user_id=user_id,
                submitted_photos=submitted_photos,
                successful_photos=successful_pairs * 2,
                submitted_pairs=submitted_pairs,
                successful_pairs=successful_pairs,
                error=error,
            )
        except Exception:
            log.exception("failed to record isolation upload statistics")

    def get_statistics(self, *, dataset_id: str | None = None) -> IsolationStatistics:
        if dataset_id and not isolation_db.get_dataset(self.db_path, dataset_id):
            raise KeyError(dataset_id)
        raw = isolation_db.get_statistics(self.db_path, dataset_id=dataset_id)
        from datetime import datetime, timezone

        for contributor in raw["contributors"]:
            ts = contributor.get("last_submitted_at")
            contributor["last_submitted_at"] = (
                datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                if ts is not None
                else None
            )
        return IsolationStatistics(dataset_id=dataset_id, **raw)

    def export_dataset_zip(self, dataset_id: str, dest_zip: Path | None = None) -> Path:
        row = isolation_db.get_dataset(self.db_path, dataset_id)
        if not row:
            raise KeyError(dataset_id)
        ddir = self._dataset_dir(dataset_id)
        if dest_zip is None:
            export_root = self.root_dir / "uploads" / "isolation" / "exports"
            export_root.mkdir(parents=True, exist_ok=True)
            dest_zip = export_root / f"{dataset_id}.zip"
        dest_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in ddir.rglob("*"):
                if path.is_file():
                    zf.write(path, arcname=str(path.relative_to(ddir)).replace("\\", "/"))
        return dest_zip

    def start_train(
        self,
        *,
        dataset_id: str,
        base_model: str,
        epochs: int,
        val_split: float,
        force_min_pairs: bool,
        owner_user_id: str | None,
        grow_active: bool = True,
        resume_from_model_id: str | None = None,
        auto_activate: bool = True,
    ) -> IsolationTrainResponse:
        row = isolation_db.get_dataset(self.db_path, dataset_id)
        if not row:
            raise KeyError(dataset_id)
        pair_count = int(row.get("pair_count") or 0)
        if pair_count < 1:
            raise ValueError("Dataset needs at least 1 before/after couple to train.")
        if pair_count < _MIN_PAIRS_WARN and not force_min_pairs:
            raise ValueError(
                f"Dataset has only {pair_count} pairs (recommended >= {_MIN_PAIRS_WARN}). "
                "Set force_min_pairs=true to override for testing."
            )

        active = isolation_db.get_active_model(self.db_path)
        parent_id = resume_from_model_id
        if not parent_id and grow_active and active:
            parent_id = active["model_id"]

        # One growing model: reuse active model_id when grow_active is on.
        if grow_active and active and (not resume_from_model_id or resume_from_model_id == active["model_id"]):
            model_id = active["model_id"]
            parent_id = active["model_id"]
        else:
            model_id = str(uuid.uuid4())

        provider = self._train_provider()
        job_id = str(uuid.uuid4())
        isolation_db.insert_train_job(
            self.db_path,
            {
                "job_id": job_id,
                "dataset_id": dataset_id,
                "status": IsolationTrainStatus.queued.value,
                "progress": 0,
                "base_model": base_model,
                "epochs": epochs,
                "val_split": val_split,
                "model_id": model_id,
                "provider": provider,
                "owner_user_id": owner_user_id,
                "resume_from_model_id": parent_id,
                "grow_active": grow_active,
                "auto_activate": auto_activate,
            },
        )
        if provider == "api_thread":
            self._schedule_train(job_id)
        return self._train_response(job_id)

    def _train_provider(self) -> str:
        if os.getenv("APP_REMOTE_WORKERS", "").strip().lower() in ("1", "true", "yes"):
            return "worker_gpu"
        if os.getenv("ISOLATION_TRAIN_ON_API", "").strip().lower() in ("1", "true", "yes"):
            return "api_thread"
        return "worker_gpu"

    def get_train(self, job_id: str) -> IsolationTrainResponse | None:
        row = isolation_db.get_train_job(self.db_path, job_id)
        if not row:
            return None
        return self._train_response(job_id, row=row)

    def _train_response(self, job_id: str, row: dict | None = None) -> IsolationTrainResponse:
        row = row or isolation_db.get_train_job(self.db_path, job_id)
        assert row is not None
        metrics_raw = isolation_db.metrics_from_json(row.get("metrics_json"))
        metrics = None
        if metrics_raw:
            allowed = set(IsolationTrainMetrics.model_fields)
            metrics = IsolationTrainMetrics(**{k: metrics_raw[k] for k in metrics_raw if k in allowed})
        generation = None
        if metrics_raw and metrics_raw.get("generation") is not None:
            try:
                generation = int(metrics_raw["generation"])
            except (TypeError, ValueError):
                generation = None
        return IsolationTrainResponse(
            job_id=row["job_id"],
            status=IsolationTrainStatus(row["status"]),
            progress=int(row.get("progress") or 0),
            metrics=metrics,
            model_id=row.get("model_id"),
            error=row.get("error"),
            generation=generation,
            resumed_from=row.get("resume_from_model_id"),
        )

    def claim_train_for_worker(self) -> dict | None:
        row = isolation_db.claim_next_train_job(self.db_path)
        if not row:
            return None
        dataset_id = row["dataset_id"]
        zip_path = self.export_dataset_zip(dataset_id)
        public = os.getenv("APP_PUBLIC_BASE_URL", "").strip().rstrip("/")
        rel = zip_path.relative_to(self.root_dir).as_posix()
        dataset_zip_url = f"{public}/{rel}" if public else f"/{rel}"

        resume_id = row.get("resume_from_model_id")
        resume_checkpoint_url = None
        if resume_id:
            ckpt = self._model_dir(str(resume_id)) / "checkpoint.pt"
            if ckpt.is_file():
                # Auth'd worker endpoint (models/ is not a public static mount).
                resume_checkpoint_url = (
                    f"{public}/internal/worker/isolation/models/{resume_id}/checkpoint"
                    if public
                    else f"/internal/worker/isolation/models/{resume_id}/checkpoint"
                )

        return {
            "kind": "isolation_train",
            "job_id": row["job_id"],
            "dataset_id": dataset_id,
            "model_id": row.get("model_id"),
            "base_model": row.get("base_model"),
            "epochs": row.get("epochs"),
            "val_split": row.get("val_split"),
            "dataset_zip_url": dataset_zip_url,
            "dataset_dir": str(self._dataset_dir(dataset_id)),
            "resume_from_model_id": resume_id,
            "resume_checkpoint_url": resume_checkpoint_url,
            "grow_active": bool(row.get("grow_active", 1)),
            "auto_activate": bool(row.get("auto_activate", 1)),
        }

    def complete_train_from_worker(
        self,
        *,
        job_id: str,
        model_id: str,
        onnx_bytes: bytes,
        metrics: dict[str, Any],
        checkpoint_bytes: bytes | None = None,
    ) -> IsolationTrainResponse:
        row = isolation_db.get_train_job(self.db_path, job_id)
        if not row:
            raise KeyError(job_id)
        model_dir = self._model_dir(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = model_dir / "model.onnx"
        onnx_path.write_bytes(onnx_bytes)
        if checkpoint_bytes:
            (model_dir / "checkpoint.pt").write_bytes(checkpoint_bytes)
        metrics_path = model_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        rel = onnx_path.relative_to(self.root_dir).as_posix()
        auto_activate = bool(row.get("auto_activate", 1))
        isolation_db.upsert_model(
            self.db_path,
            {
                "model_id": model_id,
                "name": f"isolation-{model_id[:8]}",
                "dataset_id": row["dataset_id"],
                "onnx_rel_path": rel,
                "metrics_json": json.dumps(metrics),
                "is_active": 1 if auto_activate else 0,
                "owner_user_id": row.get("owner_user_id"),
            },
        )
        if auto_activate:
            isolation_db.set_active_model(self.db_path, model_id)
            invalidate_session()
        isolation_db.update_train_job(
            self.db_path,
            job_id,
            status=IsolationTrainStatus.completed.value,
            progress=100,
            model_id=model_id,
            metrics_json=json.dumps(metrics),
            error=None,
        )
        return self._train_response(job_id)

    def fail_train(self, job_id: str, error: str) -> IsolationTrainResponse:
        isolation_db.update_train_job(
            self.db_path,
            job_id,
            status=IsolationTrainStatus.failed.value,
            error=error,
        )
        return self._train_response(job_id)

    def import_model(
        self,
        *,
        name: str,
        onnx_bytes: bytes,
        dataset_id: str | None,
        metrics: dict[str, Any] | None,
        owner_user_id: str | None,
        activate: bool = False,
    ) -> IsolationModelSummary:
        model_id = str(uuid.uuid4())
        model_dir = self._model_dir(model_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = model_dir / "model.onnx"
        onnx_path.write_bytes(onnx_bytes)
        rel = onnx_path.relative_to(self.root_dir).as_posix()
        metrics_json = json.dumps(metrics or {})
        isolation_db.insert_model(
            self.db_path,
            {
                "model_id": model_id,
                "name": name,
                "dataset_id": dataset_id,
                "onnx_rel_path": rel,
                "metrics_json": metrics_json,
                "is_active": 1 if activate else 0,
                "owner_user_id": owner_user_id,
            },
        )
        if activate:
            isolation_db.set_active_model(self.db_path, model_id)
            invalidate_session()
        return self._model_summary(isolation_db.get_model(self.db_path, model_id) or {})

    def list_models(self) -> list[IsolationModelSummary]:
        return [self._model_summary(r) for r in isolation_db.list_models(self.db_path)]

    def get_active_model(self) -> IsolationModelSummary | None:
        row = isolation_db.get_active_model(self.db_path)
        return self._model_summary(row) if row else None

    def activate_model(self, model_id: str) -> IsolationModelSummary:
        row = isolation_db.set_active_model(self.db_path, model_id)
        if not row:
            raise KeyError(model_id)
        invalidate_session()
        return self._model_summary(row)

    def _model_summary(self, row: dict[str, Any]) -> IsolationModelSummary:
        from datetime import datetime, timezone

        metrics_raw = isolation_db.metrics_from_json(row.get("metrics_json"))
        metrics = IsolationTrainMetrics(**metrics_raw) if metrics_raw else None
        created = datetime.fromtimestamp(float(row["created_at"]), tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        return IsolationModelSummary(
            model_id=row["model_id"],
            name=row["name"],
            dataset_id=row.get("dataset_id"),
            is_active=bool(row.get("is_active")),
            metrics=metrics,
            created_at=created,
        )

    def resolve_model_path(self, model_id: str | None) -> Path:
        if model_id:
            row = isolation_db.get_model(self.db_path, model_id)
            if not row:
                raise KeyError(model_id)
        else:
            row = isolation_db.get_active_model(self.db_path)
            if not row:
                raise RuntimeError("No active isolation model. Train or import a model first.")
        rel = str(row["onnx_rel_path"])
        path = self.root_dir / rel.replace("/", os.sep)
        if not path.is_file():
            raise FileNotFoundError(f"Model ONNX missing: {path}")
        return path

    def predict(
        self,
        *,
        user_id: str | None,
        image_bytes: bytes,
        model_id: str | None,
        return_json: bool,
    ) -> tuple[bytes | None, dict[str, Any] | None, int]:
        predict_id = str(uuid.uuid4())
        charge_isolation_predict(self.db_path, user_id=user_id, predict_id=predict_id)
        model_path = self.resolve_model_path(model_id)
        rgba_png, mask_png, latency_ms = predict_isolated_png(image_bytes, model_path=model_path)
        log.info(
            "isolation predict user=%s model=%s predict_id=%s latency_ms=%d",
            user_id or "anonymous",
            model_id or "active",
            predict_id,
            latency_ms,
        )
        if not return_json:
            return rgba_png, None, latency_ms

        out_dir = self._predict_root() / predict_id
        out_dir.mkdir(parents=True, exist_ok=True)
        isolated_path = out_dir / "isolated.png"
        mask_path = out_dir / "mask.png"
        isolated_path.write_bytes(rgba_png)
        mask_path.write_bytes(mask_png)
        rel_iso = isolated_path.relative_to(self.root_dir).as_posix()
        rel_mask = mask_path.relative_to(self.root_dir).as_posix()
        active = isolation_db.get_active_model(self.db_path) if not model_id else isolation_db.get_model(
            self.db_path, model_id
        )
        body = {
            "isolated_url": self._asset_url_for(rel_iso),
            "mask_url": self._asset_url_for(rel_mask),
            "model_id": (active or {}).get("model_id") or model_id or "",
            "latency_ms": latency_ms,
        }
        return None, body, latency_ms

    def get_quota(self, user_id: str):
        return get_isolation_quota(self.db_path, user_id)

    def health(self) -> IsolationHealthResponse:
        from .isolation_inference import get_or_load_session, onnxruntime_available

        active = isolation_db.get_active_model(self.db_path)
        loaded = False
        active_id = None
        active_path = None
        ort_ok = onnxruntime_available()
        if active and ort_ok:
            try:
                path = self.root_dir / str(active["onnx_rel_path"]).replace("/", os.sep)
                get_or_load_session(path)
                loaded = path.is_file()
                active_id = active["model_id"]
                active_path = str(path)
            except Exception:
                loaded = False
        return IsolationHealthResponse(
            onnxruntime_ok=ort_ok,
            model_loaded=loaded,
            active_model_id=active_id,
            active_model_path=active_path,
        )

    def _schedule_train(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._active_train:
                return
            self._active_train.add(job_id)

        def _run() -> None:
            try:
                self._run_train_job(job_id)
            finally:
                with self._lock:
                    self._active_train.discard(job_id)

        threading.Thread(target=_run, name=f"iso-train-{job_id[:8]}", daemon=True).start()

    def _run_train_job(self, job_id: str) -> None:
        from .isolation_train import run_training_job

        row = isolation_db.get_train_job(self.db_path, job_id)
        if not row:
            return
        isolation_db.update_train_job(
            self.db_path, job_id, status=IsolationTrainStatus.running.value, progress=10
        )
        try:
            dataset_dir = self._dataset_dir(row["dataset_id"])
            model_id = str(row["model_id"])
            out_dir = self._model_dir(model_id)
            resume_ckpt = None
            parent = row.get("resume_from_model_id")
            if parent:
                cand = self._model_dir(str(parent)) / "checkpoint.pt"
                if cand.is_file():
                    resume_ckpt = cand
            metrics = run_training_job(
                dataset_dir=dataset_dir,
                output_dir=out_dir,
                base_model=str(row.get("base_model") or "isnet-general-use"),
                epochs=int(row.get("epochs") or 20),
                val_split=float(row.get("val_split") or 0.2),
                resume_checkpoint=resume_ckpt,
                progress_callback=lambda p: isolation_db.update_train_job(
                    self.db_path, job_id, progress=p
                ),
            )
            # Stamp lineage onto checkpoint
            ckpt_path = out_dir / "checkpoint.pt"
            ckpt_bytes = ckpt_path.read_bytes() if ckpt_path.is_file() else None
            if ckpt_bytes:
                try:
                    import torch

                    blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                    if isinstance(blob, dict):
                        blob["model_id"] = model_id
                        torch.save(blob, str(ckpt_path))
                        ckpt_bytes = ckpt_path.read_bytes()
                except Exception:
                    pass
            onnx_path = out_dir / "model.onnx"
            self.complete_train_from_worker(
                job_id=job_id,
                model_id=model_id,
                onnx_bytes=onnx_path.read_bytes(),
                metrics=metrics,
                checkpoint_bytes=ckpt_bytes,
            )
        except Exception as exc:
            log.exception("isolation train %s failed", job_id)
            self.fail_train(job_id, str(exc))


class IsolationQuotaExceeded(IsolationQuotaExceededError):
    pass
