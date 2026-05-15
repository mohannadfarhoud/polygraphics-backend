"""Phase 4 of the protocol: write a COLMAP-style sparse reconstruction.

We emit the **text** flavour (``cameras.txt`` / ``images.txt`` / ``points3D.txt``).
Both COLMAP itself and the official ``gaussian-splatting`` data reader fall back
to text when the ``.bin`` files are missing, so this is a portable seed.

Layout produced:

    <scene_dir>/
        images/                       (input photos symlinked or copied)
        sparse/0/cameras.txt
        sparse/0/images.txt
        sparse/0/points3D.txt
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from .multiview_scene import MultiviewMetricScene


def _rotation_matrix_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a unit quaternion (w, x, y, z) — COLMAP order."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(max(1e-12, 1.0 + R[0, 0] - R[1, 1] - R[2, 2]))
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(max(1e-12, 1.0 + R[1, 1] - R[0, 0] - R[2, 2]))
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(max(1e-12, 1.0 + R[2, 2] - R[0, 0] - R[1, 1]))
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    n = np.sqrt(w * w + x * x + y * y + z * z) or 1.0
    return float(w / n), float(x / n), float(y / n), float(z / n)


def write_colmap_text(
    scene: MultiviewMetricScene,
    *,
    scene_dir: Path,
    images_subdir: str = "images",
    sparse_subdir: str = "sparse/0",
    max_points: int = 200_000,
) -> Path:
    """Write a minimal COLMAP text reconstruction from a metric multi-view scene.

    Returns the path of the ``sparse/0`` directory.
    """
    images_dir = scene_dir / images_subdir
    sparse_dir = scene_dir / sparse_subdir
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    image_names: list[str] = []
    for src in scene.image_paths:
        dst = images_dir / src.name
        if not dst.exists():
            shutil.copyfile(str(src), str(dst))
        image_names.append(src.name)

    cameras_path = sparse_dir / "cameras.txt"
    with cameras_path.open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for i, K in enumerate(scene.intrinsics, start=1):
            W, H = scene.image_sizes[i - 1]
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])
            f.write(f"{i} PINHOLE {W} {H} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f}\n")

    images_path = sparse_dir / "images.txt"
    with images_path.open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for i, w2c in enumerate(scene.poses_w2c, start=1):
            R = np.asarray(w2c[:3, :3], dtype=np.float64)
            t = np.asarray(w2c[:3, 3], dtype=np.float64).reshape(3)
            qw, qx, qy, qz = _rotation_matrix_to_quaternion(R)
            name = image_names[i - 1]
            f.write(
                f"{i} {qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} "
                f"{t[0]:.8f} {t[1]:.8f} {t[2]:.8f} {i} {name}\n"
            )
            # No 2D track is emitted — gaussian-splatting tolerates the empty line.
            f.write("\n")

    pts = scene.points
    rgb = scene.colors
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]
        rgb = rgb[idx]

    points_path = sparse_dir / "points3D.txt"
    with points_path.open("w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for pid, (xyz, color) in enumerate(zip(pts, rgb), start=1):
            r, g, b = int(color[0]), int(color[1]), int(color[2])
            f.write(
                f"{pid} {float(xyz[0]):.6f} {float(xyz[1]):.6f} {float(xyz[2]):.6f} "
                f"{r} {g} {b} 0.0\n"
            )

    return sparse_dir
