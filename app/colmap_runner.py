"""COLMAP sparse I/O utilities.

The mesh pipeline uses **MapAnything** for metric multi-view geometry. This module keeps COLMAP-era
utilities for sparse **binary** conversion and ``points3D.txt`` parsing (Gaussian Splatting CPU fallback).

Legacy ``run_colmap_sparse*`` helpers remain for uncommon manual calls but expect a compat
``colmap_binary_path`` attribute on settings when used.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .runtime_settings import RuntimeSettings


@dataclass
class ColmapSparseResult:
    points_xyz: np.ndarray  # (N, 3) float32
    colors_rgb: np.ndarray  # (N, 3) uint8
    cameras: list  # list[color_baking.CameraView]; deferred import to avoid cycle


def run_colmap_sparse(
    masked_images: list[Path],
    settings: RuntimeSettings,
    *,
    workspace: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if len(masked_images) < 2:
        raise ValueError("COLMAP needs at least 2 images")

    colmap_bin = (getattr(settings, "colmap_binary_path", None) or "").strip()
    if not colmap_bin or not Path(colmap_bin).exists():
        raise RuntimeError(
            "Classic COLMAP SfM is no longer wired to the mesh path; use reconstruction_backend "
            "`mapanything`. If you reached this intentionally, supply colmap_binary_path on RuntimeSettings "
            "(legacy field)."
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

    fe_base_args = [
        colmap_bin,
        "feature_extractor",
        "--database_path",
        str(db_path),
        "--image_path",
        str(images_dir),
        "--ImageReader.single_camera",
        "1",
    ]
    want_gpu = bool(getattr(settings, "colmap_sift_gpu", True))
    if want_gpu:
        # COLMAP CLI changed across versions:
        # - newer builds: --FeatureExtraction.use_gpu
        # - older builds: --SiftExtraction.use_gpu
        gpu_variants = (
            fe_base_args + ["--FeatureExtraction.use_gpu", "1"],
            fe_base_args + ["--SiftExtraction.use_gpu", "1"],
        )
        for i, args in enumerate(gpu_variants):
            try:
                _run(args, "feature_extractor")
                break
            except RuntimeError as exc:
                msg = str(exc)
                if "unrecognised option" in msg and i + 1 < len(gpu_variants):
                    continue
                if "unrecognised option" in msg and i + 1 == len(gpu_variants):
                    _run(fe_base_args, "feature_extractor")
                    break
                raise
    else:
        _run(fe_base_args, "feature_extractor")
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


def run_colmap_sparse_with_cameras(
    masked_images: list[Path],
    settings: RuntimeSettings,
    *,
    workspace: Path,
) -> ColmapSparseResult:
    """Like :func:`run_colmap_sparse` but also returns per-view cameras (intrinsics + w2c)."""
    points, colors = run_colmap_sparse(masked_images, settings, workspace=workspace)

    sparse_dir = (workspace.resolve() / "sparse")
    sub_dirs = [d for d in sparse_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if not sub_dirs:
        return ColmapSparseResult(points, colors, [])
    chosen = max(sub_dirs, key=_count_points)
    txt_dir = chosen.parent / f"{chosen.name}_txt"
    if not (txt_dir / "cameras.txt").is_file() or not (txt_dir / "images.txt").is_file():
        return ColmapSparseResult(points, colors, [])

    try:
        cameras = _read_colmap_cameras_views(
            cameras_txt=txt_dir / "cameras.txt",
            images_txt=txt_dir / "images.txt",
            masked_images=masked_images,
        )
    except Exception:
        cameras = []
    return ColmapSparseResult(points, colors, cameras)


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


def read_points3d_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Public alias for sparse-point parsing (used by Gaussian Splatting CPU fallback)."""
    return _read_points3d_txt(path)


def _quat_xyzw_to_rotmat(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """COLMAP stores quaternions as (qw, qx, qy, qz). Return a 3x3 rotation matrix."""
    n = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n == 0:
        return np.eye(3, dtype=np.float64)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _read_colmap_cameras_views(
    *,
    cameras_txt: Path,
    images_txt: Path,
    masked_images: list[Path],
) -> list:
    """Parse COLMAP ``cameras.txt`` + ``images.txt`` into a list[CameraView] aligned to ``masked_images``."""
    from .color_baking import CameraView  # local import to avoid cycle

    # cameras.txt: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    cams: dict[int, dict] = {}
    with cameras_txt.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cam_id = int(parts[0])
                model = parts[1]
                w = int(parts[2])
                h = int(parts[3])
                params = [float(p) for p in parts[4:]]
            except ValueError:
                continue
            cams[cam_id] = {"model": model, "w": w, "h": h, "params": params}

    # images.txt: each image is two lines; the first holds pose+camera_id+name, second holds keypoints (skip).
    name_to_pose: dict[str, dict] = {}
    expect_meta = True
    with images_txt.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if expect_meta:
                parts = line.split()
                if len(parts) < 10:
                    continue
                try:
                    qw, qx, qy, qz = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
                    tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                    cam_id = int(parts[8])
                    name = parts[9]
                except ValueError:
                    continue
                name_to_pose[name] = {
                    "qw": qw, "qx": qx, "qy": qy, "qz": qz,
                    "tx": tx, "ty": ty, "tz": tz,
                    "cam_id": cam_id,
                }
                expect_meta = False
            else:
                expect_meta = True

    views: list = []
    for img_path in masked_images:
        meta = name_to_pose.get(img_path.name)
        if meta is None:
            continue
        cam = cams.get(meta["cam_id"])
        if cam is None:
            continue

        # Build K from supported COLMAP camera models. SIMPLE_PINHOLE/SIMPLE_RADIAL: f, cx, cy[, k];
        # PINHOLE: fx, fy, cx, cy; OPENCV: fx, fy, cx, cy[, k1, k2, p1, p2].
        params = cam["params"]
        model = cam["model"]
        try:
            if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
                fx = fy = float(params[0])
                cx, cy = float(params[1]), float(params[2])
            elif model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"):
                fx = float(params[0]); fy = float(params[1])
                cx = float(params[2]); cy = float(params[3])
            else:
                fx = fy = float(params[0])
                cx, cy = cam["w"] / 2.0, cam["h"] / 2.0
        except (IndexError, ValueError):
            continue

        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        R = _quat_xyzw_to_rotmat(meta["qw"], meta["qx"], meta["qy"], meta["qz"])
        t = np.array([meta["tx"], meta["ty"], meta["tz"]], dtype=np.float64)
        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = R
        w2c[:3, 3] = t

        views.append(
            CameraView(
                image_path=img_path,
                image_size=(int(cam["w"]), int(cam["h"])),
                K=K,
                w2c=w2c,
            )
        )
    return views


def load_sparse_points_from_gs_scene(scene_dir: Path, settings: RuntimeSettings) -> tuple[np.ndarray, np.ndarray]:
    """Read COLMAP sparse 3D points + RGB from a gaussian-splatting ``scene_dir``.

    Prefer ``sparse/0/points3D.txt`` (MapAnything/COLMAP-text bridge). If only binary models exist,
    optionally run ``model_converter`` when ``colmap_binary_path`` is set on settings.
    """
    scene_dir = scene_dir.resolve()
    direct_txt = scene_dir / "sparse" / "0" / "points3D.txt"
    if direct_txt.is_file():
        return _read_points3d_txt(direct_txt)

    colmap_bin = (getattr(settings, "colmap_binary_path", None) or "").strip()
    if not colmap_bin or not Path(colmap_bin).exists():
        raise RuntimeError(
            "sparse/points3D.txt missing and COLMAP CLI unavailable: install COLMAP temporarily and set "
            "colmap_binary_path to convert binaries, or rebuild the GS scene via MapAnything (writes text)."
        )

    sparse_root = scene_dir / "sparse"
    sub_dirs = [d for d in sparse_root.iterdir() if d.is_dir()]
    if not sub_dirs:
        raise RuntimeError(f"No COLMAP sparse reconstruction under {sparse_root}")

    chosen = max(sub_dirs, key=_count_points)
    txt_dir = chosen.parent / f"{chosen.name}_txt_export"
    txt_dir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        [
            colmap_bin,
            "model_converter",
            "--input_path",
            str(chosen),
            "--output_path",
            str(txt_dir),
            "--output_type",
            "TXT",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-20:]
        raise RuntimeError("COLMAP model_converter failed:\n" + "\n".join(tail))

    return _read_points3d_txt(txt_dir / "points3D.txt")
