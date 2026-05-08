"""COLMAP sparse Structure-from-Motion → colored point cloud (mesh-path input).

This module is used by ``Dust3RReconstructor`` when the resolved backend is
``colmap`` (which happens when ``reconstruction_backend == "colmap"`` or when
``"auto"`` selects COLMAP for jobs with many photos).

What it does
------------
1. Copies the input (masked) images into ``<workspace>/images/``.
2. Runs the standard COLMAP pipeline:
     ``feature_extractor`` → ``exhaustive_matcher`` → ``mapper``.
3. Converts the binary sparse model to text (``model_converter``) so we can
   parse ``points3D.txt`` without depending on the binary format.
4. Returns ``(points Nx3 float32, colors Nx3 uint8)`` — RGB sampled by COLMAP
   from the original images, exactly what we need to color the mesh.

Notes
-----
- Works on Windows with both ``colmap.exe`` and a wrapper ``COLMAP.bat``.
- Sparse-only: dense reconstruction needs CUDA on Windows. The sparse cloud
  is enough for a reasonable Poisson surface; per-vertex color is what we
  want for the GLB.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from .runtime_settings import RuntimeSettings


def run_colmap_sparse(
    masked_images: list[Path],
    settings: RuntimeSettings,
    *,
    workspace: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if len(masked_images) < 2:
        raise ValueError("COLMAP needs at least 2 images")

    colmap_bin = (settings.colmap_binary_path or "").strip()
    if not colmap_bin or not Path(colmap_bin).exists():
        raise RuntimeError(
            "COLMAP backend requires a valid colmap_binary_path "
            "(set it via PUT /settings, e.g. C:\\COLMAP\\COLMAP.bat)."
        )

    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    images_dir = workspace / "images"
    sparse_dir = workspace / "sparse"
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Stage images in the workspace (COLMAP requires a directory).
    for src in masked_images:
        dst = images_dir / src.name
        if not dst.exists():
            shutil.copyfile(str(src), str(dst))

    db_path = workspace / "database.db"
    if db_path.exists():
        try:
            db_path.unlink()
        except OSError:
            pass

    def _run(args: list[str], step: str) -> None:
        proc = subprocess.run(args, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-30:]
            raise RuntimeError(
                f"COLMAP step failed ({step}): {' '.join(args)}\n" + "\n".join(tail)
            )

    _run(
        [
            colmap_bin, "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--ImageReader.single_camera", "1",
        ],
        "feature_extractor",
    )
    _run(
        [
            colmap_bin, "exhaustive_matcher",
            "--database_path", str(db_path),
        ],
        "exhaustive_matcher",
    )
    _run(
        [
            colmap_bin, "mapper",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir),
        ],
        "mapper",
    )

    sub_dirs = [d for d in sparse_dir.iterdir() if d.is_dir()]
    if not sub_dirs:
        raise RuntimeError(
            "COLMAP mapper produced no reconstruction. Make sure your photos overlap "
            "and have enough texture (no plain backgrounds), then try again."
        )
    # COLMAP can output multiple submodels (0/, 1/, ...); use the densest one.
    chosen = max(sub_dirs, key=_count_points)

    txt_dir = chosen.parent / f"{chosen.name}_txt"
    txt_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            colmap_bin, "model_converter",
            "--input_path", str(chosen),
            "--output_path", str(txt_dir),
            "--output_type", "TXT",
        ],
        "model_converter",
    )

    points, colors = _read_points3d_txt(txt_dir / "points3D.txt")
    if points.shape[0] < 16:
        raise RuntimeError(
            f"COLMAP reconstruction has only {points.shape[0]} 3D points; "
            "not enough to mesh. Add more / better-textured photos."
        )
    return points, colors


def _count_points(model_dir: Path) -> int:
    """Cheap heuristic to pick the largest sub-model: file size of points3D.bin or .txt."""
    for name in ("points3D.bin", "points3D.txt"):
        p = model_dir / name
        if p.is_file():
            try:
                return p.stat().st_size
            except OSError:
                return 0
    return 0


def _read_points3d_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a COLMAP ``points3D.txt`` into (xyz float32, rgb uint8)."""
    if not path.is_file():
        raise RuntimeError(f"Missing COLMAP points3D file: {path}")

    xyz: list[tuple[float, float, float]] = []
    rgb: list[tuple[int, int, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # Format: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]...
            if len(parts) < 7:
                continue
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
            except ValueError:
                continue
            xyz.append((x, y, z))
            rgb.append((r, g, b))

    if not xyz:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    return (
        np.asarray(xyz, dtype=np.float32),
        np.asarray(rgb, dtype=np.uint8),
    )
