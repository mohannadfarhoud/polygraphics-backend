from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class SurfaceRegionEntry:
    view_name: str
    source_path: Path
    output_path: Path
    area_ratio: float
    confidence: float
    centroid: tuple[float, float]
    label_hint: str | None = None
    failed: bool = False
    failure_reason: str | None = None


@dataclass
class SurfaceRegionResult:
    remap: dict[str, Path]
    entries: list[SurfaceRegionEntry]


def _largest_component(mask_u8: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return mask_u8
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask_u8)
    out[labels == best] = 255
    return out


def _extract_dominant_surface_mask(
    rgb: np.ndarray,
    *,
    smooth_percentile: float,
    min_area_ratio: float,
    expand_px: int,
) -> np.ndarray:
    fg = np.any(rgb > 8, axis=2).astype(np.uint8) * 255
    if not np.any(fg):
        return fg

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)

    vals = mag[fg > 0]
    if vals.size < 64:
        return fg
    p = float(np.clip(smooth_percentile, 1.0, 99.0))
    th = float(np.percentile(vals, p))
    smooth = ((mag <= th).astype(np.uint8) * 255)
    cand = cv2.bitwise_and(smooth, fg)

    k = max(3, int(round(min(rgb.shape[:2]) * 0.01)))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, kernel)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, kernel)
    cand = _largest_component(cand)

    min_ratio = float(np.clip(min_area_ratio, 0.01, 1.0))
    area = float(np.count_nonzero(cand))
    fg_area = float(np.count_nonzero(fg))
    if fg_area <= 1 or (area / fg_area) < min_ratio:
        return fg

    ex = max(0, int(expand_px))
    if ex > 0:
        ek = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ex + 1, 2 * ex + 1))
        cand = cv2.dilate(cand, ek, iterations=1)
        cand = cv2.bitwise_and(cand, fg)

    return cand


def _score_region_confidence(
    rgb: np.ndarray,
    fg: np.ndarray,
    region_mask: np.ndarray,
) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    sharpness = float(np.var(lap)) / 600.0
    sharpness = float(np.clip(sharpness, 0.0, 1.0))

    fg_count = float(max(1, np.count_nonzero(fg)))
    area_ratio = float(np.count_nonzero(region_mask)) / fg_count
    area_score = float(np.clip((area_ratio - 0.02) / 0.18, 0.0, 1.0))

    ys, xs = np.where(region_mask > 0)
    if ys.size == 0:
        center_score = 0.0
    else:
        h, w = gray.shape[:2]
        cx = float(xs.mean()) / max(1.0, float(w - 1))
        cy = float(ys.mean()) / max(1.0, float(h - 1))
        center_dist = float(np.hypot(cx - 0.5, cy - 0.5))
        center_score = float(np.clip(1.0 - (center_dist / 0.70), 0.0, 1.0))

    smooth_score = float(np.clip(1.0 - sharpness * 0.7, 0.0, 1.0))
    score = 0.40 * area_score + 0.35 * smooth_score + 0.25 * center_score
    return float(np.clip(score, 0.0, 1.0))


def _region_centroid(region_mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(region_mask > 0)
    if ys.size == 0:
        return 0.5, 0.5
    h, w = region_mask.shape[:2]
    cx = float(xs.mean()) / max(1.0, float(w - 1))
    cy = float(ys.mean()) / max(1.0, float(h - 1))
    return cx, cy


def infer_label_from_centroid(cx: float, cy: float) -> str:
    if cy < 0.30:
        return "top"
    if cx < 0.33:
        return "left"
    if cx > 0.67:
        return "right"
    return "front"


def build_dominant_surface_texture_images(
    *,
    job_id: str,
    image_paths: list[Path],
    cache_root: Path,
    smooth_percentile: float,
    min_area_ratio: float,
    expand_px: int,
) -> SurfaceRegionResult:
    out_dir = cache_root / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    remap: dict[str, Path] = {}
    entries: list[SurfaceRegionEntry] = []
    seen: set[Path] = set()
    idx = 0
    for src in image_paths:
        p = src.resolve()
        if p in seen:
            continue
        seen.add(p)

        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            entries.append(
                SurfaceRegionEntry(
                    view_name=p.name,
                    source_path=p,
                    output_path=p,
                    area_ratio=0.0,
                    confidence=0.0,
                    centroid=(0.5, 0.5),
                    failed=True,
                    failure_reason="image_read_failed",
                )
            )
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        fg = (np.any(rgb > 8, axis=2).astype(np.uint8) * 255)
        m = _extract_dominant_surface_mask(
            rgb,
            smooth_percentile=smooth_percentile,
            min_area_ratio=min_area_ratio,
            expand_px=expand_px,
        )
        out_rgb = np.zeros_like(rgb)
        out_rgb[m > 0] = rgb[m > 0]
        dst = out_dir / f"surface_tex_{idx:03d}.png"
        idx += 1
        cv2.imwrite(str(dst), cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))

        fg_count = float(max(1, np.count_nonzero(fg)))
        area_ratio = float(np.count_nonzero(m)) / fg_count
        confidence = _score_region_confidence(rgb, fg, m)
        cx, cy = _region_centroid(m)
        label_hint = infer_label_from_centroid(cx, cy)

        entries.append(
            SurfaceRegionEntry(
                view_name=p.name,
                source_path=p,
                output_path=dst,
                area_ratio=area_ratio,
                confidence=confidence,
                centroid=(cx, cy),
                label_hint=label_hint,
                failed=False,
                failure_reason=None,
            )
        )

        remap[str(p)] = dst
        remap[str(src)] = dst
        remap[src.name] = dst

    return SurfaceRegionResult(remap=remap, entries=entries)

