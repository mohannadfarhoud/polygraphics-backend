from __future__ import annotations

import os
from pathlib import Path

from .runtime_settings import RuntimeSettings, effective_reconstruction_backend


def assert_pipeline_ready(settings: RuntimeSettings) -> None:
    """
    When allow_placeholder_pipeline is False, require checkpoints and backends
    so jobs fail fast instead of producing meaningless meshes.
    """
    if settings.allow_placeholder_pipeline:
        return

    if not settings.skip_sam_segmentation:
        backend = str(getattr(settings, "isolation_backend", "sam")).strip().lower()
        if backend == "sam":
            if not settings.sam_checkpoint_path or not Path(settings.sam_checkpoint_path).is_file():
                raise RuntimeError(
                    "SAM isolation requires sam_checkpoint_path to an existing .pth on the machine "
                    "that runs the pipeline (see README). On a remote GPU worker use POLYGRAPH_OVERRIDE_SAM_CHECKPOINT "
                    "in .env.worker — PUT /settings paths refer to the API host, not the worker disk. "
                    "Or switch isolation_backend to rembg. "
                    "With skip_sam_segmentation=true SAM/rembg is bypassed (upload RGBA cutouts)."
                )
        elif backend == "rembg":
            try:
                import rembg  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "rembg isolation backend is selected but rembg is not installed. "
                    "Install with: pip install rembg onnxruntime pillow"
                ) from exc
        else:
            raise RuntimeError(f"Unknown isolation_backend={backend!r}; expected 'sam' or 'rembg'.")

    def _need_mapanything() -> None:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "MapAnything reconstruction needs PyTorch installed on the worker. "
                "Install torch + CUDA for GPU jobs. Original error: " + str(exc)
            ) from exc
        try:
            import mapanything  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "MapAnything Python package missing. pip install git+https://github.com/facebookresearch/map-anything.git "
                "(see README). Original error: " + str(exc)
            ) from exc

    def _need_dust3r() -> None:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "DUSt3R reconstruction needs PyTorch installed on the worker. "
                "Install torch + CUDA for GPU jobs. Original error: " + str(exc)
            ) from exc
        try:
            import dust3r  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "DUSt3R Python package missing. Install naver/dust3r in this environment. "
                "Original error: " + str(exc)
            ) from exc
        if not settings.dust3r_checkpoint_path or not Path(settings.dust3r_checkpoint_path).exists():
            raise RuntimeError(
                "dust3r_checkpoint_path must point to an existing local DUSt3R checkpoint (.pth) "
                "on the machine that runs reconstruction (worker path in split deploy)."
            )

    def _need_colmap() -> None:
        colmap_bin = (getattr(settings, "colmap_binary_path", None) or "").strip()
        if not colmap_bin or not Path(colmap_bin).is_file():
            raise RuntimeError(
                "COLMAP backend selected but colmap_binary_path is missing or invalid. "
                "Set it to a valid executable/batch on the machine running reconstruction "
                '(example: "C:\\COLMAP\\COLMAP.bat").'
            )

    def _need_ai_prior() -> None:
        provider = str(getattr(settings, "ai_prior_provider", "command")).strip().lower()
        if provider == "command":
            raw = (getattr(settings, "ai_prior_command", None) or "").strip()
            if not raw:
                raise RuntimeError(
                    "AI prior backend selected but ai_prior_command is empty. "
                    "Set ai_prior_command to an executable/script that outputs a mesh."
                )
            p = Path(raw)
            if not p.is_file():
                import shutil

                if shutil.which(raw) is None:
                    raise RuntimeError(f"ai_prior_command not found: {raw}")
            key_env = str(getattr(settings, "ai_prior_api_key_env", "AI_PRIOR_API_KEY")).strip()
            key_required = bool(getattr(settings, "ai_prior_require_api_key", False))
            if key_required and not os.getenv(key_env, "").strip():
                raise RuntimeError(
                    f"AI prior provider requires API key env {key_env!r}, but it is not set on this machine."
                )
        elif provider == "mock":
            pass
        else:
            raise RuntimeError(
                f"Unsupported ai_prior_provider={provider!r}; expected 'command' or 'mock'."
            )

    backend = effective_reconstruction_backend(settings)

    if backend == "mapanything":
        _need_mapanything()
    elif backend == "dust3r":
        _need_dust3r()
    elif backend == "colmap":
        _need_colmap()
    elif backend == "gaussian_splatting":
        _need_mapanything()
        repo = Path(settings.gs_repo_path or "")
        try:
            import torch

            cuda_ok = bool(torch.cuda.is_available())
        except Exception:
            cuda_ok = False

        need_repo = cuda_ok or not settings.gs_allow_cpu_fallback
        if need_repo:
            if not settings.gs_repo_path or not repo.is_dir() or not (repo / "train.py").is_file():
                raise RuntimeError(
                    "Gaussian Splatting on GPU requires gs_repo_path to a clone of "
                    "https://github.com/graphdeco-inria/gaussian-splatting (with train.py). "
                    "CPU-only: set gs_allow_cpu_fallback=true to skip the trainer and export a colored .ply."
                )
    elif backend == "ai_prior":
        _need_ai_prior()
    elif backend == "hybrid_prior_refine":
        _need_ai_prior()
        refine_backend = str(getattr(settings, "hybrid_refine_backend", "mapanything")).strip().lower()
        if refine_backend == "mapanything":
            _need_mapanything()
        elif refine_backend == "dust3r":
            _need_dust3r()
        elif refine_backend == "colmap":
            _need_colmap()
        elif refine_backend == "none":
            pass
        else:
            raise RuntimeError(
                f"Unsupported hybrid_refine_backend={refine_backend!r}; expected mapanything/dust3r/colmap/none."
            )
