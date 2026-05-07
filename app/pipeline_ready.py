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
            "Real SAM is required: set sam_checkpoint_path to an existing .pth file (see README), "
            "or set allow_placeholder_pipeline=true only for local demos."
        )

    if settings.reconstruction_backend == "dust3r":
        ck = (settings.dust3r_checkpoint_path or "").strip()
        if not ck:
            raise RuntimeError(
                "Real DUSt3R is required: set dust3r_checkpoint_path (local checkpoint folder/file "
                "or Hugging Face model id), install dust3r + torch (see README), "
                "or set allow_placeholder_pipeline=true only for local demos."
            )
    elif settings.reconstruction_backend == "colmap":
        if not settings.colmap_binary_path or not Path(settings.colmap_binary_path).exists():
            raise RuntimeError(
                "COLMAP backend selected: set colmap_binary_path to the COLMAP executable, "
                "or switch reconstruction_backend to dust3r."
            )
    elif settings.reconstruction_backend == "gaussian_splatting":
        repo = Path(settings.gs_repo_path or "")
        if not settings.gs_repo_path or not repo.is_dir() or not (repo / "train.py").is_file():
            raise RuntimeError(
                "Gaussian Splatting backend requires gs_repo_path to point at a clone of "
                "https://github.com/graphdeco-inria/gaussian-splatting (with train.py)."
            )
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
