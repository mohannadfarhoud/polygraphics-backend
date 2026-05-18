"""Flatten RGBA cutout uploads for MapAnything when SAM masking is skipped.

The default pipeline runs **Segment Anything first** on full-frame photos (``skip_sam_segmentation=false``).
This module is only for **pre-cut** assets: every file must be **RGBA** with a **real alpha matte**
(transparent or soft-edge background). Plain RGB / JPEG / grayscale uploads are rejected so we never
silently pass scene background into MapAnything.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _alpha_indicates_cutout(alpha_u8: np.ndarray) -> bool:
    """True when alpha shows transparency or meaningful feathering (not a flat opaque sheet)."""
    a = alpha_u8.astype(np.float32) / 255.0
    fr_trans = float(np.mean(a < 0.15))
    fr_partial = float(np.mean((a > 0.05) & (a < 0.95)))
    fr_not_opaque = float(np.mean(a < 0.98))
    return fr_trans >= 0.001 or fr_partial >= 0.002 or fr_not_opaque >= 0.005


def prepare_precut_opaque_views_for_mapanything(
    job_id: str,
    image_paths: list[Path],
    *,
    masked_dir: Path,
    flatten_gray_0_255: int,
) -> list[Path]:
    """Write ``masked_<idx>.png`` composites — **RGBA cutouts only** (no SAM step).

    Per pixel: ``flatten_gray`` neutral grey where alpha≈0, otherwise RGB from the cutout.
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

        if raw.ndim == 2:
            raise RuntimeError(
                f"skip_sam_segmentation requires RGBA cutouts (alpha channel). Grayscale has no alpha: {src}. "
                "Set skip_sam_segmentation=false so Segment Anything removes the background first."
            )
        if raw.ndim != 3:
            raise RuntimeError(f"Unsupported image shape {raw.shape}: {src}")

        if raw.shape[2] != 4:
            raise RuntimeError(
                f"skip_sam_segmentation requires RGBA images (4 channels) with transparency; "
                f"got {raw.shape[2]} channels on {src}. "
                "For full-camera photos, set skip_sam_segmentation=false — SAM runs as the first pipeline step."
            )

        alpha_plane = raw[:, :, 3]
        if not _alpha_indicates_cutout(alpha_plane):
            raise RuntimeError(
                f"Alpha on {src} looks fully opaque — not a transparent-background cutout. "
                "Export RGBA with transparency, or set skip_sam_segmentation=false for automatic SAM masking."
            )

        bgra = raw
        bgr = bgra[:, :, :3].astype(np.float32)
        alpha = (bgra[:, :, 3:4].astype(np.float32) / 255.0).clip(0.0, 1.0)
        matte_bgr = np.full_like(bgr, matte, dtype=np.float32)
        comp = alpha * bgr + (1.0 - alpha) * matte_bgr
        dest = out_job / f"masked_{idx:03d}.png"
        cv2.imwrite(str(dest), comp.astype(np.uint8))
        outputs.append(dest)

    return outputs
