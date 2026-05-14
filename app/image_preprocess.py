"""Downscale uploaded photos before SAM / COLMAP / GS so 4K phone images do not blow RAM or VRAM."""

from __future__ import annotations

from pathlib import Path


def downscale_job_images_if_needed(
    job_id: str,
    image_paths: list[Path],
    base_dir: Path,
    max_longest_side: int,
) -> list[Path]:
    """Return paths to images whose longest side is at most ``max_longest_side``.

    Larger images are resized with aspect ratio preserved (``INTER_AREA``) and written under
    ``base_dir / job_id /``. Images already within the bound are returned unchanged.
    Unreadable files are passed through unchanged.
    """
    if max_longest_side < 1:
        return list(image_paths)

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV (cv2) is required for image preprocessing.") from exc

    result: list[Path] = []
    out_dir: Path | None = None

    for i, p in enumerate(image_paths):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            result.append(p)
            continue
        h, w = img.shape[:2]
        m = max(h, w)
        if m <= max_longest_side:
            result.append(p)
            continue

        if out_dir is None:
            out_dir = base_dir / job_id
            out_dir.mkdir(parents=True, exist_ok=True)

        scale = max_longest_side / float(m)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        ext = p.suffix if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff") else ".jpg"
        dest = out_dir / f"input_{i:03d}{ext}"
        if ext.lower() in (".jpg", ".jpeg"):
            cv2.imwrite(str(dest), resized, (cv2.IMWRITE_JPEG_QUALITY, 92))
        else:
            cv2.imwrite(str(dest), resized)
        result.append(dest)

    return result
