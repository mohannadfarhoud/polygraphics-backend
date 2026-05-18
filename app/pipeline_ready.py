from __future__ import annotations

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
        if not settings.sam_checkpoint_path or not Path(settings.sam_checkpoint_path).is_file():
            raise RuntimeError(
                "Real SAM is required: sam_checkpoint_path must point to an existing .pth on the machine "
                "that runs the pipeline (see README). On a remote GPU worker use POLYGRAPH_OVERRIDE_SAM_CHECKPOINT "
                "in .env.worker — PUT /settings paths refer to the API host, not the worker disk. "
                "Or set allow_placeholder_pipeline=true only for local demos. "
                "With skip_sam_segmentation=true SAM is not loaded — uploads must be RGBA cutouts with transparency."
            )

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

    backend = effective_reconstruction_backend(settings)

    if backend == "mapanything":
        _need_mapanything()
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
