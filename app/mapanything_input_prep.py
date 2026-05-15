"""Flatten RGBA uploads to opaque PNGs for MapAnything when SAM masking is skipped."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def prepare_precut_opaque_views_for_mapanything(
    job_id: str,
    image_paths: list[Path],
    *,
    masked_dir: Path,
    flatten_gray_0_255: int,
) -> list[Path]:
    """Write ``masked_<idx>.png`` (or copy passthrough RGB) — no SAM step.

    * **RGBA PNG / WebP** (fourth channel alpha): opaque composite with ``flatten_gray``
      as RGB fill where alpha≈0. Output is three-channel PNG (BGR in memory, saved as PNG).
    * **RGB / grayscale**: copied as-is next to filenames ``masked_<idx>.<ext>`` (extensions preserved).

    ``flatten_gray`` is reproduced on all RGB channels before OpenCV saves (neutral gray backdrop).
    """
    import cv2

    matte = float(np.clip(int(flatten_gray_0_255), 0, 255))
    out_job = masked_dir / job_id
    out_job.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for idx, src in enumerate(image_paths):
        raw = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise RuntimeError(f"Unreadable upload for skip-SAM pipeline: {src}")

        fallback_ext = ".png"
        suffix = src.suffix.lower() if src.suffix else fallback_ext

        if raw.ndim == 2:
            bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
            dest = out_job / f"masked_{idx:03d}{suffix if suffix != '' else fallback_ext}"
            cv2.imwrite(str(dest), bgr)
            outputs.append(dest)
            continue

        if raw.ndim != 3:
            raise RuntimeError(f"Unsupported image shape {raw.shape}: {src}")

        channels = raw.shape[2]

        if channels == 4:
            bgra = raw
            bgr = bgra[:, :, :3].astype(np.float32)
            alpha = (bgra[:, :, 3:4].astype(np.float32) / 255.0).clip(0.0, 1.0)
            matte_bgr = np.full_like(bgr, matte, dtype=np.float32)
            comp = alpha * bgr + (1.0 - alpha) * matte_bgr
            dest = out_job / f"masked_{idx:03d}.png"
            cv2.imwrite(str(dest), comp.astype(np.uint8))
            outputs.append(dest)
        elif channels == 3:
            dest = out_job / f"masked_{idx:03d}{suffix if suffix != '' else fallback_ext}"
            cv2.imwrite(str(dest), raw)
            outputs.append(dest)
        else:
            raise RuntimeError(f"Need 1-, 3-, or 4-channel image, got channels={channels}: {src}")

    return outputs
