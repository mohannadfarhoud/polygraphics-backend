"""
Gaussian Splatting integration.

Real path:
  1. Build a COLMAP scene from masked images (cameras + sparse points), or import points from DUSt3R.
  2. Run training via the official `gaussian-splatting` repo:
     https://github.com/graphdeco-inria/gaussian-splatting
     `python <gs_repo_path>/train.py -s <scene_dir> -m <model_dir> --iterations N --sh_degree D --resolution R`
  3. Copy the resulting `point_cloud/iteration_<N>/point_cloud.ply` to `output/<job_id>.ply`.

Placeholder path (allow_placeholder_pipeline=True):
  - Writes a tiny GS-style PLY with a few thousand white Gaussians at random positions
    so the wiring works end-to-end on machines without a GPU. NOT real geometry.

Notes:
  - GS training officially requires a CUDA GPU. Some forks support CPU/Metal but are slow.
  - This runner shells out to `gs_python_executable` (or current interpreter) to allow a
    separate environment with torch+cuda installed.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

from .runtime_settings import RuntimeSettings


def run_gaussian_splatting(
    job_id: str,
    masked_images: list[Path],
    work_dir: Path,
    output_ply: Path,
    settings: RuntimeSettings,
) -> Path:
    """Train a Gaussian Splatting scene and write `output_ply`. Returns its path."""
    if len(masked_images) < 2:
        raise ValueError("Gaussian Splatting needs at least 2 images")

    work_dir.mkdir(parents=True, exist_ok=True)
    output_ply.parent.mkdir(parents=True, exist_ok=True)

    if settings.allow_placeholder_pipeline:
        _write_placeholder_gs_ply(output_ply, n_points=8000)
        return output_ply

    repo = Path(settings.gs_repo_path or "")
    if not repo.is_dir() or not (repo / "train.py").is_file():
        raise RuntimeError(
            "gs_repo_path must point to a clone of "
            "https://github.com/graphdeco-inria/gaussian-splatting (where train.py lives)."
        )

    scene_dir = work_dir / "scene"
    model_dir = work_dir / "model"
    scene_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    if settings.gs_init_source == "colmap":
        _build_colmap_scene(masked_images, scene_dir, settings)
    elif settings.gs_init_source == "dust3r":
        _build_dust3r_scene(masked_images, scene_dir, settings)
    else:
        raise RuntimeError(f"Unknown gs_init_source {settings.gs_init_source!r}")

    py = settings.gs_python_executable or sys.executable
    cmd = [
        py,
        str(repo / "train.py"),
        "-s",
        str(scene_dir),
        "-m",
        str(model_dir),
        "--iterations",
        str(settings.gs_iterations),
        "--sh_degree",
        str(settings.gs_sh_degree),
    ]
    if settings.gs_resolution and settings.gs_resolution > 0:
        cmd += ["--resolution", str(settings.gs_resolution)]

    proc = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-30:]
        raise RuntimeError(
            "Gaussian Splatting training failed.\n"
            f"Command: {' '.join(cmd)}\n"
            "Last stderr/stdout lines:\n" + "\n".join(tail)
        )

    final_ply = _find_latest_ply(model_dir)
    if final_ply is None:
        raise RuntimeError(f"GS training finished but no point_cloud.ply found under {model_dir}")

    shutil.copyfile(str(final_ply), str(output_ply))
    return output_ply


def _find_latest_ply(model_dir: Path) -> Path | None:
    pc_root = model_dir / "point_cloud"
    if not pc_root.is_dir():
        plys = list(model_dir.rglob("point_cloud.ply"))
        return max(plys, key=lambda p: p.stat().st_mtime) if plys else None

    iters = [d for d in pc_root.iterdir() if d.is_dir() and d.name.startswith("iteration_")]
    if not iters:
        return None
    iters.sort(key=lambda d: int(d.name.split("_", 1)[1]) if d.name.split("_", 1)[1].isdigit() else -1)
    candidate = iters[-1] / "point_cloud.ply"
    return candidate if candidate.is_file() else None


def _build_colmap_scene(masked_images: list[Path], scene_dir: Path, settings: RuntimeSettings) -> None:
    """
    Run COLMAP CLI to produce a scene the gaussian-splatting trainer can consume.
    Layout produced:
      scene_dir/
        images/                      (copied input images)
        sparse/0/{cameras,images,points3D}.bin
    """
    colmap = settings.colmap_binary_path
    if not colmap or not Path(colmap).exists():
        raise RuntimeError(
            "GS init=colmap requires colmap_binary_path. "
            "Install COLMAP and set its executable path in settings."
        )

    images_dir = scene_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    for src in masked_images:
        dst = images_dir / src.name
        if not dst.exists():
            shutil.copyfile(str(src), str(dst))

    db_path = scene_dir / "database.db"
    sparse_dir = scene_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    def _run(args: list[str]) -> None:
        proc = subprocess.run(args, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-20:]
            raise RuntimeError(f"COLMAP step failed: {' '.join(args)}\n" + "\n".join(tail))

    _run([colmap, "feature_extractor", "--database_path", str(db_path), "--image_path", str(images_dir),
          "--ImageReader.single_camera", "1"])
    _run([colmap, "exhaustive_matcher", "--database_path", str(db_path)])
    _run([colmap, "mapper", "--database_path", str(db_path),
          "--image_path", str(images_dir), "--output_path", str(sparse_dir)])

    sub = [d for d in sparse_dir.iterdir() if d.is_dir()]
    if not sub:
        raise RuntimeError(
            "COLMAP mapper produced no reconstruction. "
            "Check that the input photos have enough texture and overlap."
        )


def _build_dust3r_scene(masked_images: list[Path], scene_dir: Path, settings: RuntimeSettings) -> None:
    raise RuntimeError(
        "GS init=dust3r is scaffolded but not yet implemented. "
        "Use gs_init_source='colmap' for now."
    )


def _write_placeholder_gs_ply(path: Path, n_points: int = 8000) -> None:
    """
    Minimal Gaussian-Splatting PLY (binary little-endian) with the standard fields the
    official viewers expect:
      x y z nx ny nz f_dc_0..2 f_rest_0..44 opacity scale_0..2 rot_0..3
    Values are dummy (white, small isotropic Gaussians) — for wiring tests only.
    """
    rng = np.random.default_rng(42)
    xyz = (rng.standard_normal((n_points, 3)) * 0.5).astype(np.float32)
    nrm = np.zeros((n_points, 3), dtype=np.float32)
    f_dc = np.full((n_points, 3), 0.5, dtype=np.float32)         # SH DC ~ light grey
    f_rest = np.zeros((n_points, 45), dtype=np.float32)           # 3 * 15 (SH degree 3)
    opacity = np.full((n_points, 1), -2.0, dtype=np.float32)      # logit(~0.12)
    scale = np.full((n_points, 3), -3.0, dtype=np.float32)        # log scale (~0.05)
    rot = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n_points, 1))

    rows = np.concatenate([xyz, nrm, f_dc, f_rest, opacity, scale, rot], axis=1).astype(np.float32)

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n_points}",
        "property float x", "property float y", "property float z",
        "property float nx", "property float ny", "property float nz",
        "property float f_dc_0", "property float f_dc_1", "property float f_dc_2",
        *[f"property float f_rest_{i}" for i in range(45)],
        "property float opacity",
        "property float scale_0", "property float scale_1", "property float scale_2",
        "property float rot_0", "property float rot_1", "property float rot_2", "property float rot_3",
        "end_header",
        "",
    ]
    header = "\n".join(header_lines).encode("ascii")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header)
        f.write(rows.tobytes(order="C"))
        # silence unused warning for struct
        _ = struct.calcsize("f")
