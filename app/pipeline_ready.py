from __future__ import annotations

from pathlib import Path

from .runtime_settings import RuntimeSettings


def assert_pipeline_ready(settings: RuntimeSettings) -> None:
    """
    When allow_placeholder_pipeline is False, require checkpoints and backends
    so jobs fail fast instead of producing meaningless meshes.
    """
    if settings.allow_placeholder_pipeline:
        return

    if not settings.sam_checkpoint_path or not Path(settings.sam_checkpoint_path).is_file():
        raise RuntimeError(
            "Real SAM is required: sam_checkpoint_path must point to an existing .pth on the machine "
            "that runs the pipeline (see README). On a remote GPU worker use POLYGRAPH_OVERRIDE_SAM_CHECKPOINT "
            "in .env.worker — PUT /settings paths refer to the API host, not the worker disk. "
            "Or set allow_placeholder_pipeline=true only for local demos."
        )

    backend = settings.reconstruction_backend

    def _need_dust3r() -> None:
        ck = (settings.dust3r_checkpoint_path or "").strip()
        if not ck:
            raise RuntimeError(
                "Real DUSt3R is required: set dust3r_checkpoint_path (local checkpoint folder/file "
                "or Hugging Face model id), install dust3r + torch (see README), "
                "or set allow_placeholder_pipeline=true only for local demos."
            )

    def _need_colmap() -> None:
        if not settings.colmap_binary_path or not Path(settings.colmap_binary_path).exists():
            raise RuntimeError(
                "COLMAP backend selected: set colmap_binary_path to the COLMAP executable "
                "(e.g. C:\\COLMAP\\COLMAP.bat), or switch reconstruction_backend to dust3r."
            )

    if backend == "dust3r":
        _need_dust3r()
    elif backend == "colmap":
        _need_colmap()
    elif backend == "auto":
        # Auto picks dust3r for small jobs and colmap for larger ones, so both
        # need to be available; we don't know the image count at this point.
        _need_dust3r()
        _need_colmap()
    elif backend == "gaussian_splatting":
        repo = Path(settings.gs_repo_path or "")
        try:
            import torch

            cuda_ok = bool(torch.cuda.is_available())
        except Exception:
            cuda_ok = False

        if settings.gs_init_source == "colmap":
            if not settings.colmap_binary_path or not Path(settings.colmap_binary_path).exists():
                raise RuntimeError(
                    "Gaussian Splatting with gs_init_source='colmap' requires colmap_binary_path."
                )
        elif settings.gs_init_source == "dust3r":
            if not (settings.dust3r_checkpoint_path or "").strip():
                raise RuntimeError(
                    "Gaussian Splatting with gs_init_source='dust3r' requires dust3r_checkpoint_path."
                )

        need_repo = cuda_ok or not settings.gs_allow_cpu_fallback
        if need_repo:
            if not settings.gs_repo_path or not repo.is_dir() or not (repo / "train.py").is_file():
                raise RuntimeError(
                    "Gaussian Splatting on GPU requires gs_repo_path to a clone of "
                    "https://github.com/graphdeco-inria/gaussian-splatting (with train.py). "
                    "CPU-only: set gs_allow_cpu_fallback=true to skip the trainer and export a colored .ply."
                )
