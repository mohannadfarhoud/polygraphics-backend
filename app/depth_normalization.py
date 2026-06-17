from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .runtime_settings import RuntimeSettings


@dataclass
class DepthNormalizationResult:
    masked_paths: list[Path]
    original_paths: list[Path]
    selected_masked_path: Path | None
    consistency_score: float
    report: dict
    applied: bool


def _mask_area_ratio(path: Path) -> float:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return 0.0
    fg = np.any(img > 8, axis=2)
    return float(np.count_nonzero(fg)) / float(max(1, fg.size))


def _sharpness_and_brightness(path: Path) -> tuple[float, float]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return 0.0, 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright = float(gray.mean())
    return blur, bright


def _select_reference_index(
    *,
    valid_indices: list[int],
    depth_proxy: list[float],
    quality: list[float],
    mode: str,
) -> int:
    if not valid_indices:
        return 0
    mode = mode.strip().lower()
    if mode == "first":
        return 0 if 0 in valid_indices else valid_indices[0]
    vals = np.array([depth_proxy[i] for i in valid_indices], dtype=np.float64)
    med = float(np.median(vals))
    # Pick frame near median depth while preferring better image quality.
    best_idx = valid_indices[0]
    best_score = float("inf")
    for i in valid_indices:
        proximity = abs(depth_proxy[i] - med) / max(med, 1e-6)
        s = proximity - (0.15 * quality[i])
        if s < best_score:
            best_score = s
            best_idx = i
    return int(best_idx)


def run_depth_normalization_gate(
    *,
    job_id: str,
    masked_paths: list[Path],
    original_paths: list[Path],
    settings: RuntimeSettings,
    upload_dir: Path,
) -> DepthNormalizationResult:
    enabled = bool(getattr(settings, "depth_normalization_enabled", True))
    if (not enabled) or len(masked_paths) < 2:
        report = {
            "job_id": job_id,
            "enabled": enabled,
            "applied": False,
            "reason": "disabled_or_too_few_images",
            "consistency_score": 1.0,
            "selected_input": masked_paths[0].name if masked_paths else None,
            "kept_files": [p.name for p in masked_paths],
        }
        return DepthNormalizationResult(
            masked_paths=list(masked_paths),
            original_paths=list(original_paths),
            selected_masked_path=(masked_paths[0] if masked_paths else None),
            consistency_score=1.0,
            report=report,
            applied=False,
        )

    max_relative = float(getattr(settings, "depth_consistency_max_relative", 0.35))
    min_kept = int(getattr(settings, "depth_consistency_min_kept_images", 3))
    apply_filter = bool(getattr(settings, "depth_consistency_apply_filter", True))
    mode = str(getattr(settings, "depth_reference_mode", "median_best"))
    w_quality = float(getattr(settings, "depth_selection_quality_weight", 0.55))
    w_proximity = float(getattr(settings, "depth_selection_proximity_weight", 0.45))

    blur_min = float(getattr(settings, "capture_blur_min", 35.0))
    bmin = float(getattr(settings, "capture_brightness_min", 20.0))
    bmax = float(getattr(settings, "capture_brightness_max", 235.0))

    n = min(len(masked_paths), len(original_paths))
    area_ratio: list[float] = []
    depth_proxy: list[float] = []
    quality: list[float] = []
    valid_indices: list[int] = []
    for i in range(n):
        a = _mask_area_ratio(masked_paths[i])
        area_ratio.append(a)
        # Proxy depth from object footprint: closer objects occupy larger area.
        depth_proxy.append(float(1.0 / max(np.sqrt(max(a, 1e-6)), 1e-6)))
        blur, bright = _sharpness_and_brightness(original_paths[i])
        blur_q = min(1.0, max(0.0, blur / max(blur_min * 2.0, 1.0)))
        exp_q = 1.0 - min(1.0, abs(bright - (bmin + bmax) * 0.5) / max((bmax - bmin) * 0.5, 1.0))
        q = float((0.6 * blur_q) + (0.4 * exp_q))
        quality.append(q)
        if a > 1e-6:
            valid_indices.append(i)

    ref_idx = _select_reference_index(
        valid_indices=valid_indices,
        depth_proxy=depth_proxy,
        quality=quality,
        mode=mode,
    )
    ref_depth = depth_proxy[ref_idx] if ref_idx < len(depth_proxy) else 1.0

    rel_dev: list[float] = []
    kept_indices: list[int] = []
    rejected_indices: list[int] = []
    for i in range(n):
        d = abs(depth_proxy[i] - ref_depth) / max(ref_depth, 1e-6)
        rel_dev.append(float(d))
        if i == ref_idx or d <= max_relative:
            kept_indices.append(i)
        else:
            rejected_indices.append(i)

    if not apply_filter or len(kept_indices) < max(2, min_kept):
        kept_indices = list(range(n))
        rejected_indices = []

    proximity_scores = [1.0 - min(1.0, rel_dev[i] / max(max_relative, 1e-6)) for i in range(n)]
    selection_scores = [
        float((w_quality * quality[i]) + (w_proximity * proximity_scores[i]))
        for i in range(n)
    ]
    kept_set = set(kept_indices)
    selected_idx = max(kept_indices, key=lambda i: selection_scores[i]) if kept_indices else ref_idx

    if kept_indices:
        mean_dev_kept = float(np.mean([min(1.0, rel_dev[i] / max(max_relative * 1.5, 1e-6)) for i in kept_indices]))
        consistency = float(max(0.0, min(1.0, (1.0 - mean_dev_kept) * (len(kept_indices) / max(1, n)))))
    else:
        consistency = 0.0

    norm_masked = [masked_paths[i] for i in kept_indices]
    norm_original = [original_paths[i] for i in kept_indices]
    selected_masked = masked_paths[selected_idx] if 0 <= selected_idx < n else (masked_paths[0] if masked_paths else None)

    per_view = []
    for i in range(n):
        per_view.append(
            {
                "index": i,
                "file": masked_paths[i].name,
                "area_ratio": round(float(area_ratio[i]), 6),
                "depth_proxy": round(float(depth_proxy[i]), 6),
                "quality_score": round(float(quality[i]), 6),
                "relative_deviation": round(float(rel_dev[i]), 6),
                "selection_score": round(float(selection_scores[i]), 6),
                "is_reference": bool(i == ref_idx),
                "kept": bool(i in kept_set),
                "is_selected_input": bool(i == selected_idx),
            }
        )

    report = {
        "job_id": job_id,
        "enabled": enabled,
        "applied": True,
        "reference_mode": mode,
        "reference_index": int(ref_idx),
        "reference_file": masked_paths[ref_idx].name if 0 <= ref_idx < n else None,
        "selected_input_index": int(selected_idx),
        "selected_input": selected_masked.name if selected_masked else None,
        "consistency_score": round(float(consistency), 4),
        "thresholds": {
            "max_relative_deviation": max_relative,
            "min_kept_images": min_kept,
            "apply_filter": apply_filter,
            "selection_quality_weight": w_quality,
            "selection_proximity_weight": w_proximity,
        },
        "total_count": n,
        "kept_count": len(kept_indices),
        "kept_files": [masked_paths[i].name for i in kept_indices],
        "rejected_files": [masked_paths[i].name for i in rejected_indices],
        "per_view": per_view,
    }

    out = upload_dir / job_id / "depth_normalization_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return DepthNormalizationResult(
        masked_paths=norm_masked,
        original_paths=norm_original,
        selected_masked_path=selected_masked,
        consistency_score=consistency,
        report=report,
        applied=True,
    )
