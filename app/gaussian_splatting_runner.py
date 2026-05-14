"""
Gaussian Splatting integration.

GPU path (CUDA + official repo):
  1. Build a COLMAP scene from masked images, or DUSt3R → COLMAP-text bridge.
  2. Run ``python <gs_repo_path>/train.py`` from graphdeco-inria/gaussian-splatting.
  3. Copy ``point_cloud/iteration_<N>/point_cloud.ply`` to ``output/<job_id>.ply``.

CPU path (``gs_allow_cpu_fallback=true``, default): no ``train.py``, no CUDA extensions.
  After step (1), export a valid 3DGS-format ``.ply`` from sparse colored points — real RGB,
  one Gaussian per point — viewable in splat viewers but **not** neural optimization.

Placeholder path (allow_placeholder_pipeline=True):
  Random Gaussians for wiring tests only.

Notes:
  - Official train.py requires NVIDIA CUDA + built submodules (see scripts/install_gaussian_splatting_windows.ps1).
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from .cuda_memory import purge_torch_cuda
from .runtime_settings import RuntimeSettings
from .colmap_runner import load_sparse_points_from_gs_scene
from .gs_ply_export import write_gaussian_ply_from_colored_points

ProgressCallback = Callable[[str, int], None]

# Official train.py resets opacity every `--opacity_reset_interval` steps. On *short* runs (common
# on 8 GB GPUs, e.g. 5000 iters) that reset mid-run can leave zero splats after pruning, and the CUDA
# rasterizer then fails in backward with invalid gradient shapes — graphdeco-inria/gaussian-splatting#482.
_GS_SHORT_RUN_MAX_ITERS = 10_000


def _effective_opacity_reset_interval(settings: RuntimeSettings) -> int:
    """Avoid scheduling opacity reset inside a short training run when it would still fire."""
    iters = int(settings.gs_iterations)
    interval = int(settings.gs_opacity_reset_interval)
    if interval <= 0:
        return interval
    if iters > _GS_SHORT_RUN_MAX_ITERS:
        return interval
    if interval < iters:
        return iters + 1
    return interval


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _vacuum_cuda_cache() -> None:
    """Free Python-held CUDA allocations before spawning ``train.py`` (SAM+DUSt3R then GS)."""
    purge_torch_cuda()


def _emit(progress_callback: ProgressCallback | None, stage: str, progress: int) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(stage, progress)
    except Exception:
        # Progress callbacks must never break the pipeline.
        pass


def run_gaussian_splatting(
    job_id: str,
    masked_images: list[Path],
    work_dir: Path,
    output_ply: Path,
    settings: RuntimeSettings,
    *,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    """Train a Gaussian Splatting scene and write `output_ply`. Returns its path."""
    if len(masked_images) < 2:
        raise ValueError("Gaussian Splatting needs at least 2 images")

    work_dir.mkdir(parents=True, exist_ok=True)
    output_ply.parent.mkdir(parents=True, exist_ok=True)

    if settings.allow_placeholder_pipeline:
        _write_placeholder_gs_ply(output_ply, n_points=8000)
        return output_ply

    cuda_ok = _torch_cuda_available()
    cpu_fallback = (not cuda_ok) and settings.gs_allow_cpu_fallback

    if not cuda_ok and not settings.gs_allow_cpu_fallback:
        raise RuntimeError(
            "No CUDA GPU detected for Gaussian Splatting training. Install NVIDIA CUDA + "
            "graphdeco-inria/gaussian-splatting (see scripts/install_gaussian_splatting_windows.ps1), "
            "or set gs_allow_cpu_fallback=true to export a colored CPU .ply from sparse points "
            "(no neural optimization)."
        )

    repo = Path(settings.gs_repo_path or "")
    if cuda_ok:
        if not repo.is_dir() or not (repo / "train.py").is_file():
            raise RuntimeError(
                "GPU Gaussian Splatting requires gs_repo_path pointing to "
                "https://github.com/graphdeco-inria/gaussian-splatting (with train.py)."
            )

    scene_dir = work_dir / "scene"
    model_dir = work_dir / "model"
    scene_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    if settings.gs_init_source == "colmap":
        _emit(progress_callback, "phase_4_colmap_scene", 50)
        _build_colmap_scene(masked_images, scene_dir, settings)
    elif settings.gs_init_source == "dust3r":
        _emit(progress_callback, "phase_2_alignment", 45)
        _build_dust3r_scene(
            masked_images,
            scene_dir,
            settings,
            progress_callback=progress_callback,
        )
    else:
        raise RuntimeError(f"Unknown gs_init_source {settings.gs_init_source!r}")

    _vacuum_cuda_cache()

    _emit(progress_callback, "phase_5_gaussian_splatting", 65)

    if cpu_fallback:
        xyz, rgb = load_sparse_points_from_gs_scene(scene_dir, settings)
        if xyz.shape[0] < 8:
            raise RuntimeError(
                "Too few sparse 3D points for CPU Gaussian export. Try more overlapping photos "
                "or gs_init_source=dust3r."
            )
        write_gaussian_ply_from_colored_points(
            xyz,
            rgb,
            output_ply,
            max_points=int(settings.gs_cpu_max_points),
        )
        return output_ply

    py = settings.gs_python_executable or sys.executable
    opacity_reset = _effective_opacity_reset_interval(settings)
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
        "--opacity_reset_interval",
        str(opacity_reset),
    ]
    if settings.gs_resolution and settings.gs_resolution > 0:
        cmd += ["--resolution", str(settings.gs_resolution)]
    densify_until = int(getattr(settings, "gs_densify_until_iter", 0))
    if densify_until > 0:
        cmd += ["--densify_until_iter", str(densify_until)]

    train_env = os.environ.copy()
    train_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    proc = subprocess.run(
        cmd,
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
        env=train_env,
    )
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

    fe_base_args = [
        colmap,
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
        gpu_variants = (
            fe_base_args + ["--FeatureExtraction.use_gpu", "1"],
            fe_base_args + ["--SiftExtraction.use_gpu", "1"],
        )
        for i, args in enumerate(gpu_variants):
            try:
                _run(args)
                break
            except RuntimeError as exc:
                msg = str(exc)
                if "unrecognised option" in msg and i + 1 < len(gpu_variants):
                    continue
                if "unrecognised option" in msg and i + 1 == len(gpu_variants):
                    _run(fe_base_args)
                    break
                raise
    else:
        _run(fe_base_args)
    _run([colmap, "exhaustive_matcher", "--database_path", str(db_path)])
    _run([colmap, "mapper", "--database_path", str(db_path),
          "--image_path", str(images_dir), "--output_path", str(sparse_dir)])

    sub = [d for d in sparse_dir.iterdir() if d.is_dir()]
    if not sub:
        raise RuntimeError(
            "COLMAP mapper produced no reconstruction. "
            "Check that the input photos have enough texture and overlap."
        )


def _build_dust3r_scene(
    masked_images: list[Path],
    scene_dir: Path,
    settings: RuntimeSettings,
    *,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Phases 2-4 of the protocol: run DUSt3R on the masked images, sanitize the cloud,
    then write a COLMAP sparse reconstruction (text format) the gaussian-splatting
    trainer can consume.

    Layout produced::

        scene_dir/
            images/
            sparse/0/{cameras.txt, images.txt, points3D.txt}
    """
    from .colmap_bridge import write_colmap_text
    from .dust3r_runner import run_dust3r_scene

    dust3r_scene = run_dust3r_scene(masked_images, settings)
    _emit(progress_callback, "phase_3_sanitization", 55)
    _emit(progress_callback, "phase_4_colmap_bridge", 60)
    write_colmap_text(dust3r_scene, scene_dir=scene_dir)


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
