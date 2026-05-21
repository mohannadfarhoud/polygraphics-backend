from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .runtime_settings import RuntimeSettings


@dataclass
class CaptureQualityResult:
    kept_paths: list[Path]
    rejected: list[dict]
    score: float
    report: dict


def _frame_metrics(path: Path) -> tuple[float, float, np.ndarray] | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())
    tiny = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return blur, bright, tiny


def run_capture_quality_gate(
    *,
    job_id: str,
    image_paths: list[Path],
    settings: RuntimeSettings,
    upload_dir: Path,
) -> CaptureQualityResult:
    enabled = bool(getattr(settings, "capture_quality_gate_enabled", True))
    blur_min = float(getattr(settings, "capture_blur_min", 35.0))
    bmin = float(getattr(settings, "capture_brightness_min", 20.0))
    bmax = float(getattr(settings, "capture_brightness_max", 235.0))
    motion_min = float(getattr(settings, "capture_min_frame_delta", 0.010))
    policy = str(getattr(settings, "capture_reject_policy", "soft")).strip().lower()
    min_kept = int(getattr(settings, "capture_min_kept_images", 8))

    if not enabled or len(image_paths) < 3:
        report = {
            "job_id": job_id,
            "enabled": enabled,
            "applied": False,
            "reason": "disabled_or_too_few_images",
            "score": 1.0,
            "kept_count": len(image_paths),
            "total_count": len(image_paths),
            "rejected": [],
        }
        return CaptureQualityResult(list(image_paths), [], 1.0, report)

    rejected: list[dict] = []
    kept: list[Path] = []
    prev_tiny: np.ndarray | None = None
    valid_metrics = 0
    sum_score = 0.0

    for p in image_paths:
        met = _frame_metrics(p)
        if met is None:
            rejected.append({"file": p.name, "reason": "unreadable"})
            continue
        blur, bright, tiny = met
        reasons: list[str] = []
        if blur < blur_min:
            reasons.append("blur")
        if bright < bmin or bright > bmax:
            reasons.append("exposure")
        frame_delta = None
        if prev_tiny is not None:
            frame_delta = float(np.mean(np.abs(tiny - prev_tiny)))
            if frame_delta < motion_min:
                reasons.append("low_motion")
        prev_tiny = tiny

        blur_score = min(1.0, max(0.0, blur / max(blur_min * 2.0, 1.0)))
        exp_score = 1.0 - min(1.0, abs(bright - (bmin + bmax) * 0.5) / max((bmax - bmin) * 0.5, 1.0))
        mot_score = 1.0 if frame_delta is None else min(1.0, max(0.0, frame_delta / max(motion_min * 2.0, 1e-4)))
        frame_score = float((0.45 * blur_score) + (0.35 * exp_score) + (0.20 * mot_score))
        valid_metrics += 1
        sum_score += frame_score

        if reasons:
            rejected.append(
                {
                    "file": p.name,
                    "reason": "+".join(reasons),
                    "blur": round(blur, 3),
                    "brightness": round(bright, 3),
                    "frame_delta": None if frame_delta is None else round(frame_delta, 6),
                    "score": round(frame_score, 4),
                }
            )
        else:
            kept.append(p)

    score = float(sum_score / max(1, valid_metrics))
    min_kept = max(2, min_kept)

    applied = True
    if policy not in ("soft", "hard"):
        policy = "soft"

    if len(kept) < min_kept:
        if policy == "hard":
            raise RuntimeError(
                f"Capture quality gate rejected too many images ({len(kept)}/{len(image_paths)} kept, "
                f"minimum required {min_kept}). Retake with steadier motion and locked camera settings."
            )
        # Soft policy: do not drop images if the gate gets too strict.
        kept = list(image_paths)
        applied = False

    report = {
        "job_id": job_id,
        "enabled": enabled,
        "applied": applied,
        "policy": policy,
        "score": round(score, 4),
        "thresholds": {
            "blur_min": blur_min,
            "brightness_min": bmin,
            "brightness_max": bmax,
            "min_frame_delta": motion_min,
            "min_kept_images": min_kept,
        },
        "total_count": len(image_paths),
        "kept_count": len(kept),
        "rejected_count": len(rejected) if applied else 0,
        "rejected": rejected if applied else [],
    }

    out_path = upload_dir / job_id / "capture_quality_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return CaptureQualityResult(kept, rejected if applied else [], score, report)

