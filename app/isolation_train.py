"""Isolation model training (GPU worker or API thread)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .isolation_mask import mask_from_after_image

log = logging.getLogger(__name__)


def _content_hash_from_files(before: Path, after: Path | None) -> str:
    h = hashlib.sha256()
    h.update(before.read_bytes())
    h.update(b"\0")
    if after is not None and after.is_file():
        h.update(after.read_bytes())
    return h.hexdigest()


def _list_pairs(dataset_dir: Path) -> list[dict]:
    """Load training pairs, skipping exact duplicate before/after couples."""
    meta_path = dataset_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json missing in {dataset_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pairs = list(meta.get("pairs") or [])
    if not pairs:
        raise ValueError("Dataset has no pairs")
    out = []
    seen_hashes: set[str] = set()
    skipped_duplicates = 0
    for p in sorted(pairs, key=lambda x: int(x["index"])):
        idx = int(p["index"])
        pair_dir = dataset_dir / "pairs" / str(idx)
        before = next(pair_dir.glob("before.*"), None)
        after = next(pair_dir.glob("after.*"), None)
        mask = pair_dir / "mask.png"
        if not before or not mask.is_file():
            continue
        content_hash = str(p.get("content_hash") or "").strip() or _content_hash_from_files(before, after)
        if content_hash in seen_hashes:
            skipped_duplicates += 1
            log.info("skipping duplicate couple index=%s hash=%s…", idx, content_hash[:12])
            continue
        seen_hashes.add(content_hash)
        out.append({"index": idx, "before": before, "mask": mask, "content_hash": content_hash})
    if skipped_duplicates:
        log.warning("skipped %s duplicate couple(s) during training load", skipped_duplicates)
    if not out:
        raise ValueError("No valid before/mask pairs on disk (all were missing or duplicates)")
    return out


def _mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred >= 128
    gt_b = gt >= 128
    inter = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    return float(inter / union) if union > 0 else 0.0


def _baseline_rembg_mask(before_bgr: np.ndarray, base_model: str) -> np.ndarray:
    from rembg import new_session, remove

    session = new_session(base_model)
    ok, enc = cv2.imencode(".png", before_bgr)
    if not ok:
        raise RuntimeError("Failed to encode image for rembg baseline")
    out_bytes = remove(enc.tobytes(), session=session)
    arr = np.frombuffer(out_bytes, dtype=np.uint8)
    rgba = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if rgba is None or rgba.ndim < 3 or rgba.shape[2] < 4:
        raise RuntimeError("rembg baseline returned invalid RGBA")
    alpha = rgba[:, :, 3]
    _, mask = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
    return mask


def _evaluate_pairs(
    pairs: list[dict],
    *,
    base_model: str,
) -> tuple[float, float, float]:
    ious: list[float] = []
    tp = fp = fn = 0
    for item in pairs:
        before = cv2.imread(str(item["before"]), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(item["mask"]), cv2.IMREAD_GRAYSCALE)
        if before is None or gt is None:
            continue
        pred = _baseline_rembg_mask(before, base_model)
        ious.append(_mask_iou(pred, gt))
        pred_b = pred >= 128
        gt_b = gt >= 128
        tp += int(np.logical_and(pred_b, gt_b).sum())
        fp += int(np.logical_and(pred_b, ~gt_b).sum())
        fn += int(np.logical_and(~pred_b, gt_b).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    mean_iou = float(np.mean(ious)) if ious else 0.0
    return mean_iou, float(precision), float(recall)


def _rembg_session_class(base_model: str):
    from rembg.sessions import sessions_class

    for sc in sessions_class:
        try:
            if sc.name() == base_model:
                return sc
        except Exception:
            continue
    raise ValueError(f"Unknown rembg base model {base_model!r}")


def _u2net_home_candidates() -> list[Path]:
    homes: list[Path] = []
    env_home = os.getenv("U2NET_HOME", "").strip()
    if env_home:
        homes.append(Path(env_home).expanduser())
    xdg = os.getenv("XDG_DATA_HOME", "").strip()
    if xdg:
        homes.append(Path(xdg).expanduser() / ".u2net")
    homes.append(Path.home() / ".u2net")
    # de-dupe while preserving order
    out: list[Path] = []
    seen: set[str] = set()
    for h in homes:
        key = str(h.resolve()) if h.exists() else str(h)
        if key not in seen:
            seen.add(key)
            out.append(h)
    return out


def _export_rembg_onnx(base_model: str, dest: Path) -> None:
    """Download/copy rembg ONNX weights to dest (v1 export for CPU inference)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    candidates: list[Path] = []

    env_path = os.getenv("ISOLATION_BASE_ONNX_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())

    # rembg stores models as ~/.u2net/<model-name>.onnx (via pooch)
    for home in _u2net_home_candidates():
        candidates.append(home / f"{base_model}.onnx")

    try:
        sc = _rembg_session_class(base_model)
        # Forces download into U2NET_HOME and returns absolute path.
        downloaded = Path(sc.download_models())
        candidates.insert(0, downloaded)
        # Warm ORT session so subsequent eval/export paths are consistent.
        from rembg import new_session

        new_session(base_model)
    except Exception as exc:
        log.warning("rembg download/warm for %s failed: %s", base_model, exc)

    for path in candidates:
        try:
            if path.is_file() and path.stat().st_size > 1024:
                shutil.copyfile(path, dest)
                log.info("exported rembg ONNX %s -> %s (%d bytes)", path, dest, dest.stat().st_size)
                return
        except OSError:
            continue

    searched = ", ".join(str(p) for p in candidates[:8])
    raise RuntimeError(
        f"Could not locate ONNX weights for rembg model {base_model!r}. "
        f"Searched: {searched}. "
        "Install rembg with ONNX Runtime (`pip install \"rembg[cpu]\" onnxruntime`), "
        "or set ISOLATION_BASE_ONNX_PATH to a .onnx file."
    )


def _sticky_pair_split(pairs: list[dict], val_split: float) -> tuple[list[dict], list[dict]]:
    """Match finetune sticky holdout so rembg-fallback metrics stay comparable."""
    from .isolation_finetune import _pair_stable_key, _sticky_train_val_split

    keys = [_pair_stable_key(p) for p in pairs]
    train, val, _ = _sticky_train_val_split(pairs, keys=keys, val_split=val_split)
    if not val:
        val = list(train)
    return train, val


def run_training_job(
    *,
    dataset_dir: Path,
    output_dir: Path,
    base_model: str,
    epochs: int,
    val_split: float,
    progress_callback: Callable[[int], None] | None = None,
    resume_checkpoint: Path | None = None,
) -> dict:
    """Train isolation model (incremental UNet when torch is available) and write model.onnx (+ checkpoint.pt)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = _list_pairs(dataset_dir)
    train_pairs, val_pairs = _sticky_pair_split(pairs, val_split)

    if progress_callback:
        progress_callback(15)

    dev_mock = os.getenv("ISOLATION_TRAIN_DEV_MOCK", "").strip().lower() in ("1", "true", "yes")
    force_rembg = os.getenv("ISOLATION_TRAIN_REMBG_ONLY", "").strip().lower() in ("1", "true", "yes")

    # Prefer real incremental fine-tune on GPU/CPU torch when available.
    if not dev_mock and not force_rembg:
        try:
            from .isolation_finetune import torch_available, train_unet_incremental

            if torch_available():
                metrics = train_unet_incremental(
                    pairs=pairs,
                    output_dir=output_dir,
                    epochs=epochs,
                    val_split=val_split,
                    resume_checkpoint=resume_checkpoint,
                    progress_callback=progress_callback,
                )
                metrics["base_model"] = base_model
                metrics["resumed"] = bool(resume_checkpoint and Path(resume_checkpoint).is_file())
                (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                if progress_callback:
                    progress_callback(95)
                return metrics
        except Exception as exc:
            log.exception("incremental UNet fine-tune failed; falling back to rembg export: %s", exc)

    if progress_callback:
        progress_callback(20)

    if dev_mock:
        iou, precision, recall = 0.5, 0.5, 0.5
    else:
        iou, precision, recall = _evaluate_pairs(val_pairs, base_model=base_model)

    if progress_callback:
        progress_callback(60)

    onnx_dest = output_dir / "model.onnx"
    if dev_mock and not shutil.which("nvidia-smi"):
        try:
            _export_rembg_onnx(base_model, onnx_dest)
        except Exception as exc:
            log.warning("dev mock export failed (%s); writing placeholder note", exc)
            raise RuntimeError(
                "ISOLATION_TRAIN_DEV_MOCK could not export base ONNX. "
                "Install rembg or POST a prebuilt model.onnx import."
            ) from exc
    else:
        _export_rembg_onnx(base_model, onnx_dest)

    metrics = {
        "iou": round(iou, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "val_pairs": len(val_pairs),
        "train_pairs": len(train_pairs),
        "base_model": base_model,
        "epochs": epochs,
        "generation": 1,
        "backend": "rembg_export",
        "resumed": False,
        "note": (
            "Fallback rembg ONNX export (no torch fine-tune). "
            "Install torch on the GPU worker for incremental UNet training."
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if progress_callback:
        progress_callback(95)

    return metrics
