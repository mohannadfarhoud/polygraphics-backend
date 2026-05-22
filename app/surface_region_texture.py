from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


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


def build_dominant_surface_texture_images(
    *,
    job_id: str,
    image_paths: list[Path],
    cache_root: Path,
    smooth_percentile: float,
    min_area_ratio: float,
    expand_px: int,
) -> dict[str, Path]:
    out_dir = cache_root / job_id
    out_dir.mkdir(parents=True, exist_ok=True)

    remap: dict[str, Path] = {}
    seen: set[Path] = set()
    idx = 0
    for src in image_paths:
        p = src.resolve()
        if p in seen:
            continue
        seen.add(p)

        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
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

        remap[str(p)] = dst
        remap[str(src)] = dst
        remap[src.name] = dst

    return remap

