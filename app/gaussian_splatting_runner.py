"""
Gaussian Splatting integration.

GPU path (CUDA + official repo):
  1. Build COLMAP-*text* scene seed from masked images via MapAnything.
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

import json
import logging
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np

from .cuda_memory import ensure_cuda_allocator_env, purge_torch_cuda
from .runtime_settings import RuntimeSettings
from .colmap_runner import load_sparse_points_from_gs_scene
from .gs_ply_export import write_gaussian_ply_from_colored_points

ProgressCallback = Callable[[str, int], None]

_REPO_ROOT = Path(__file__).resolve().parents[1]

_logger = logging.getLogger(__name__)

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


def _assert_official_gs_train_imports(train_py: Path, py_executable: str) -> None:
    """``train.py`` imports CUDA extensions built into the interpreter that launches it (often not the API venv)."""
    proc = subprocess.run(
        [py_executable, "-c", "import diff_gaussian_rasterization, simple_knn"],
        cwd=str(train_py.parent),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return
    err = (proc.stderr or proc.stdout or "").strip()
    gs_root = train_py.parent
    third_party_root = str(gs_root.parent)
    dgr = gs_root / "submodules" / "diff-gaussian-rasterization"
    skn = gs_root / "submodules" / "simple-knn"
    raise RuntimeError(
        "Official Gaussian Splatting training needs CUDA-built Python packages in the same "
        "environment as the process that runs train.py (ModuleNotFoundError if missing).\n\n"
        f"Interpreter checked: {py_executable}\n"
        f"Gaussians repo: {gs_root}\n"
        f"Import check stderr: {err or '(empty)'}\n\n"
        "Fix (Windows, from 'x64 Native Tools Command Prompt for VS 2022' or Developer PowerShell):\n"
        "  1) Ensure submodules exist: cd the gaussian-splatting clone; "
        "git submodule update --init --recursive\n"
        "  2) Install extension wheels into this interpreter (PowerShell needs the call operator &):\n"
        f'     & "{py_executable}" -m pip install --no-build-isolation "{dgr}"\n'
        f'     & "{py_executable}" -m pip install --no-build-isolation "{skn}"\n'
        "     (In cmd.exe, omit the & and keep the quoted python path.)\n"
        "Or run (from polyGraphics-backend checkout, uses your interpreter for the submodule build):\n"
        f'  .\\scripts\\install_gaussian_splatting_windows.ps1 -SkipTorchCuda '
        f'-PythonExe "{py_executable}" -ThirdPartyRoot "{third_party_root}"'
    )


def _vacuum_cuda_cache() -> None:
    """Free Python-held CUDA allocations before spawning ``train.py`` (SAM+MapAnything then GS)."""
    purge_torch_cuda()


def _emit(progress_callback: ProgressCallback | None, stage: str, progress: int) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(stage, progress)
    except Exception:
        # Progress callbacks must never break the pipeline.
        pass


def build_gaussian_scene_workspace(
    masked_images: list[Path],
    work_dir: Path,
    settings: RuntimeSettings,
    *,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Create ``scene/`` + ``model/`` under ``work_dir`` and fill MapAnything-derived COLMAP-text for ``train.py``."""
    scene_dir = work_dir / "scene"
    model_dir = work_dir / "model"
    scene_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    _emit(progress_callback, "phase_2_alignment", 45)
    _build_mapanything_scene(
        masked_images,
        scene_dir,
        settings,
        progress_callback=progress_callback,
    )


def _swap_gs_scene_training_images_with_originals(
    scene_dir: Path,
    masked_paths: list[Path],
    original_paths: list[Path],
) -> None:
    """Replace RGB in ``scene/images`` with originals; keep filenames from ``masked_paths``.

    Pose and intrinsic metadata refer to filenames under ``images``; overwriting pixel data with
    unmasked shots fixes black-background SAM artefacts in the optimisation loss."""
    images_dir = scene_dir / "images"
    if not images_dir.is_dir():
        _logger.warning("[gs] gs_train_with_original_images: scene/%s/images missing — skip swap", scene_dir.name)
        return

    try:
        import cv2  # noqa: PLC0415
    except ImportError:
        _logger.warning("[gs] gs_train_with_original_images: OpenCV unavailable — skip swap")
        return

    n_swap = 0
    for mpath, orig in zip(masked_paths, original_paths):
        dst = images_dir / mpath.name
        if not dst.is_file():
            continue
        masked_bgr = cv2.imread(str(mpath), cv2.IMREAD_COLOR)
        rgb_bgr = cv2.imread(str(orig), cv2.IMREAD_COLOR)
        if masked_bgr is None or rgb_bgr is None:
            _logger.warning("[gs] could not read pair for swap (%s ← %s) — leaving scene image", mpath.name, orig)
            continue
        if masked_bgr.shape[:2] != rgb_bgr.shape[:2]:
            rgb_bgr = cv2.resize(
                rgb_bgr,
                (int(masked_bgr.shape[1]), int(masked_bgr.shape[0])),
                interpolation=cv2.INTER_AREA,
            )
        if not cv2.imwrite(str(dst), rgb_bgr):
            _logger.warning("[gs] failed to write swapped training image %s", dst.name)
            continue
        n_swap += 1

    if n_swap != len(masked_paths):
        _logger.warning("[gs] training-image colour swap wrote %s/%s files under scene/images/", n_swap, len(masked_paths))
    elif n_swap:
        _logger.info("[gs] replaced %s scene training images with unmasked originals (photometric supervision)", n_swap)


def run_gaussian_splatting(
    job_id: str,
    masked_images: list[Path],
    work_dir: Path,
    output_ply: Path,
    settings: RuntimeSettings,
    *,
    progress_callback: ProgressCallback | None = None,
    original_training_images: list[Path] | None = None,
) -> Path:
    """Train a Gaussian Splatting scene and write `output_ply`. Returns its path."""
    if len(masked_images) < 2:
        raise ValueError("Gaussian Splatting needs at least 2 images")

    work_dir.mkdir(parents=True, exist_ok=True)
    output_ply.parent.mkdir(parents=True, exist_ok=True)

    if settings.allow_placeholder_pipeline:
        _write_placeholder_gs_ply(output_ply, n_points=8000)
        return output_ply

    ensure_cuda_allocator_env()

    cuda_ok = _torch_cuda_available()
    cpu_fallback = (not cuda_ok) and settings.gs_allow_cpu_fallback

    if cuda_ok:
        try:
            import torch

            gpu_name = torch.cuda.get_device_name(0)
            _logger.info("[gs] CUDA OK — neural training will use train.py (%s)", gpu_name)
            print(f"[polygraph-gs] CUDA device: {gpu_name} — running train.py (not CPU fallback)", flush=True)
        except Exception:
            _logger.info("[gs] CUDA OK — neural training via train.py")
            print("[polygraph-gs] CUDA available — running train.py", flush=True)
    elif cpu_fallback:
        _logger.warning(
            "[gs] torch.cuda.is_available() is False and gs_allow_cpu_fallback=true — skipping train.py; "
            "exporting coloured PLY from sparse points only (not neural Gaussian optimisation)."
        )
        print(
            "[polygraph-gs] WARNING: no CUDA for this interpreter — Gaussian Splatting is using CPU "
            "**sparse fallback** only (no GPU train.py). Install CUDA-capable PyTorch on the worker for real splats.",
            flush=True,
        )

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

    # MapAnything can hold multi‑GB CUDA allocations. Run scene prep in a subprocess that exits before
    # ``train.py`` so VRAM is actually released on single‑GPU 8 GB boxes.
    isolate_prepare = cuda_ok and not cpu_fallback
    prep_env = os.environ.copy()

    if isolate_prepare:
        settings_json = work_dir / "_gs_prepare_settings.json"
        masked_json = work_dir / "_gs_prepare_masked_paths.json"
        settings_json.write_text(settings.model_dump_json(), encoding="utf-8")
        masked_json.write_text(json.dumps([str(p.resolve()) for p in masked_images]), encoding="utf-8")
        proc_prep = subprocess.run(
            [
                sys.executable,
                "-m",
                "worker.gs_scene_prepare",
                "--settings-json",
                str(settings_json),
                "--work-dir",
                str(work_dir),
                "--masked-json",
                str(masked_json),
            ],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            env=prep_env,
            check=False,
        )
        if proc_prep.returncode != 0:
            tail = (proc_prep.stderr or proc_prep.stdout or "").strip().splitlines()[-40:]
            raise RuntimeError(
                "Gaussian Splatting scene preparation failed (MapAnything subprocess).\n"
                + "\n".join(tail)
            )
    else:
        build_gaussian_scene_workspace(
            masked_images,
            work_dir,
            settings,
            progress_callback=progress_callback,
        )

    _vacuum_cuda_cache()

    if (
        cuda_ok
        and not cpu_fallback
        and bool(getattr(settings, "gs_train_with_original_images", True))
        and original_training_images is not None
        and len(original_training_images) == len(masked_images)
    ):
        _swap_gs_scene_training_images_with_originals(scene_dir, masked_images, original_training_images)

    _emit(progress_callback, "phase_5_gaussian_splatting", 65)

    if cpu_fallback:
        xyz, rgb = load_sparse_points_from_gs_scene(scene_dir, settings)
        if xyz.shape[0] < 8:
            raise RuntimeError(
                "Too few sparse 3D points for CPU Gaussian export. Try more overlapping photos "
                "or loosen MapAnything masking in settings."
            )
        write_gaussian_ply_from_colored_points(
            xyz,
            rgb,
            output_ply,
            max_points=int(settings.gs_cpu_max_points),
        )
        return output_ply

    py = settings.gs_python_executable or sys.executable
    train_py = repo / "train.py"
    _assert_official_gs_train_imports(train_py, py)
    opacity_reset = _effective_opacity_reset_interval(settings)
    cmd = [
        py,
        str(train_py),
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


def _build_mapanything_scene(
    masked_images: list[Path],
    scene_dir: Path,
    settings: RuntimeSettings,
    *,
    progress_callback: ProgressCallback | None = None,
) -> None:
    """Run MapAnything, then emit COLMAP text sparse reconstruction for ``train.py``."""
    from .colmap_bridge import write_colmap_text
    from .mapanything_runner import run_mapanything_scene

    scene_mv = run_mapanything_scene(masked_images, settings)
    _emit(progress_callback, "phase_3_sanitization", 55)
    _emit(progress_callback, "phase_4_colmap_bridge", 60)
    write_colmap_text(scene_mv, scene_dir=scene_dir)


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
