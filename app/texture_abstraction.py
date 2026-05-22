from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _abstract_rgb(
    rgb: np.ndarray,
    *,
    strength: float,
    detail_preserve: float,
    illumination_blur: int,
) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return rgb

    s = float(max(0.0, min(1.0, strength)))
    d = float(max(0.0, min(1.0, detail_preserve)))
    k = int(max(9, illumination_blur))
    if k % 2 == 0:
        k += 1

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Estimate low-frequency illumination and flatten it.
    l_f = l.astype(np.float32)
    illum = cv2.GaussianBlur(l_f, (k, k), 0)
    flat = (l_f / np.maximum(illum, 1.0)) * float(np.mean(illum))
    flat = np.clip(flat, 0.0, 255.0).astype(np.uint8)

    # Mild local contrast recovery on flattened luminance.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    flat_eq = clahe.apply(flat)

    # Keep high-frequency details from original luminance.
    high = cv2.subtract(l.astype(np.int16), cv2.GaussianBlur(l, (5, 5), 0).astype(np.int16))
    high = np.clip(high, -50, 50).astype(np.float32)
    l_new = flat_eq.astype(np.float32) + (d * high)
    l_new = np.clip(l_new, 0.0, 255.0).astype(np.uint8)

    out_lab = cv2.merge([l_new, a, b])
    out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

    # Blend with original to avoid over-stylization.
    blended = cv2.addWeighted(rgb, 1.0 - s, out_rgb, s, 0.0)
    return np.clip(blended, 0, 255).astype(np.uint8)


def build_abstracted_texture_images(
    *,
    job_id: str,
    image_paths: list[Path],
    cache_root: Path,
    strength: float,
    detail_preserve: float,
    illumination_blur: int,
) -> dict[str, Path]:
    out_dir = cache_root / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    seen: set[Path] = set()
    seq = 0
    for src in image_paths:
        p = src.resolve()
        if p in seen:
            continue
        seen.add(p)
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        abs_rgb = _abstract_rgb(
            rgb,
            strength=strength,
            detail_preserve=detail_preserve,
            illumination_blur=illumination_blur,
        )
        dst = out_dir / f"texture_abs_{seq:03d}.png"
        seq += 1
        cv2.imwrite(str(dst), cv2.cvtColor(abs_rgb, cv2.COLOR_RGB2BGR))
        out[str(p)] = dst
        out[str(src)] = dst
        out[src.name] = dst
    return out

