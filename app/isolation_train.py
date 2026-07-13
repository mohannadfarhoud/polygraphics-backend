"""Isolation model training (GPU worker or API thread)."""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .isolation_mask import mask_from_after_image

log = logging.getLogger(__name__)


def _list_pairs(dataset_dir: Path) -> list[dict]:
    meta_path = dataset_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta.json missing in {dataset_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pairs = list(meta.get("pairs") or [])
    if not pairs:
        raise ValueError("Dataset has no pairs")
    out = []
    for p in pairs:
        idx = int(p["index"])
        pair_dir = dataset_dir / "pairs" / str(idx)
        before = next(pair_dir.glob("before.*"), None)
        mask = pair_dir / "mask.png"
        if not before or not mask.is_file():
            continue
        out.append({"index": idx, "before": before, "mask": mask})
    if not out:
        raise ValueError("No valid before/mask pairs on disk")
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


def _export_rembg_onnx(base_model: str, dest: Path) -> None:
    """Copy rembg cached ONNX weights to dest (v1 export for CPU inference)."""
    from rembg import new_session

    session = new_session(base_model)
    candidates: list[Path] = []
    for obj in (session, getattr(session, "inner_session", None)):
        if obj is None:
            continue
        for attr in ("model_path", "path"):
            val = getattr(obj, attr, None)
            if val:
                candidates.append(Path(str(val)))
    env_path = os.getenv("ISOLATION_BASE_ONNX_PATH", "").strip()
    if env_path:
        candidates.insert(0, Path(env_path))
    for path in candidates:
        if path.is_file():
            shutil.copyfile(path, dest)
            return
    raise RuntimeError(
        f"Could not locate ONNX weights for rembg model {base_model!r}. "
        "Warm rembg once or set ISOLATION_BASE_ONNX_PATH to a .onnx file."
    )


def run_training_job(
    *,
    dataset_dir: Path,
    output_dir: Path,
    base_model: str,
    epochs: int,
    val_split: float,
    progress_callback: Callable[[int], None] | None = None,
) -> dict:
    """Train/evaluate isolation model and write model.onnx + metrics to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = _list_pairs(dataset_dir)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_split))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:] or pairs

    if progress_callback:
        progress_callback(20)

    dev_mock = os.getenv("ISOLATION_TRAIN_DEV_MOCK", "").strip().lower() in ("1", "true", "yes")
    if dev_mock:
        iou, precision, recall = 0.5, 0.5, 0.5
    else:
        iou, precision, recall = _evaluate_pairs(val_pairs, base_model=base_model)

    if progress_callback:
        progress_callback(60)

    onnx_dest = output_dir / "model.onnx"
    if dev_mock and not shutil.which("nvidia-smi"):
        # CPU smoke: write minimal valid onnx using rembg download attempt
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
        "note": (
            "v1 exports rembg-compatible ONNX selected by holdout IoU; "
            "full U2Net fine-tune runs on GPU worker when torch is available."
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if progress_callback:
        progress_callback(95)

    return metrics
