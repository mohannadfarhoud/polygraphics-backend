from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    # `dust3r`/`colmap` produce a meshed `.glb`; `gaussian_splatting` produces a `.ply` (3DGS).
    # `auto` picks DUSt3R for fewer-than-`auto_dust3r_max_images` photos and COLMAP otherwise.
    reconstruction_backend: Literal["auto", "dust3r", "colmap", "gaussian_splatting"] = "auto"
    # When reconstruction_backend == "auto", switch to COLMAP at this image count or above.
    auto_dust3r_max_images: int = Field(default=20, ge=2, le=10000)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    sam_checkpoint_path: str | None = None
    sam_model_type: str = "vit_h"
    # How to pick the foreground mask (SAM). `center_point` uses a click at the image center — best for a subject in the middle.
    # `auto_masks_center_bias` scores auto-generated masks by size × proximity to center. `auto_masks_largest_area` picks the largest mask (old behavior).
    sam_segmentation_mode: Literal[
        "center_point",
        "auto_masks_center_bias",
        "auto_masks_largest_area",
    ] = "auto_masks_center_bias"
    # AMP fp16 on CUDA for SAM forward passes (saves VRAM on e.g. RTX 3050 8GB).
    sam_use_fp16: bool = True
    dust3r_repo_path: str | None = None
    dust3r_checkpoint_path: str | None = None
    # DUSt3R global aligner (Phase 2 of the pipeline protocol).
    dust3r_aligner_iters: int = Field(default=300, ge=10, le=5000)
    dust3r_aligner_lr: float = Field(default=0.01, gt=0.0, le=1.0)
    # Drop DUSt3R points below this per-pixel confidence (0..1). 0 disables (Phase 3).
    dust3r_confidence_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    # DUSt3R inference / aligner on CUDA: mixed precision to stay under ~6–8 GB VRAM on consumer GPUs.
    dust3r_use_fp16: bool = True
    # Pair batch size for DUSt3R ``inference()`` (use 1 on RTX 3050).
    dust3r_inference_batch_size: int = Field(default=1, ge=1, le=8)
    # Internal longest-side cap for DUSt3R ``load_images`` (combined with max_image_side via min).
    dust3r_max_inference_side: int = Field(default=768, ge=256, le=8192)
    colmap_binary_path: str | None = None
    # When True, COLMAP ``feature_extractor`` gets ``--SiftExtraction.use_gpu 1`` (needs a CUDA COLMAP build).
    colmap_sift_gpu: bool = True
    # With ``reconstruction_backend == "gaussian_splatting"``, also emit mesh GLBs for A/B comparison
    # (DUSt3R vs COLMAP) before training GS: ``<job_id>_compare_dust3r.glb`` and ``<job_id>_compare_colmap.glb``.
    compare_mesh_dust3r_colmap_with_gs: bool = False
    # Gaussian Splatting (https://github.com/graphdeco-inria/gaussian-splatting)
    gs_repo_path: str | None = None
    gs_python_executable: str | None = None  # leave null to use the API's Python
    gs_init_source: Literal["colmap", "dust3r"] = "colmap"
    # ~7000 is a practical quality/speed balance on 8 GB GPUs; raise for final-quality splats.
    gs_iterations: int = Field(default=7000, ge=100, le=60000)
    gs_sh_degree: int = Field(default=3, ge=0, le=4)
    # Official train.py OptimizationParams.densify_until_iter — lower stops densification earlier (VRAM safety).
    gs_densify_until_iter: int = Field(default=5000, ge=0, le=60000)
    gs_resolution: int = Field(default=-1, ge=-1, le=8192)  # -1 = original
    # Reset Gaussian opacity every N iterations (Phase 5). vanilla default is 3000.
    gs_opacity_reset_interval: int = Field(default=3000, ge=100, le=60000)
    # When True and PyTorch sees no CUDA device, skip official train.py and write a
    # valid 3DGS-format .ply from sparse colored points (no GPU / no CUDA extensions).
    gs_allow_cpu_fallback: bool = True
    gs_cpu_max_points: int = Field(default=250_000, ge=1000, le=2_000_000)
    meshing_method: Literal["poisson", "bpa"] = "poisson"
    output_dir_name: str = "output"
    masked_dir_name: str = "masked"
    # Where raw binary masks (.png) are saved alongside the masked color images.
    masks_dir_name: str = "masks"
    # If True, also write 1-channel mask PNGs to `<masks_dir_name>/<job_id>/mask_NNN.png`.
    save_raw_masks: bool = True
    nb_neighbors: int = Field(default=20, ge=1)
    std_ratio: float = Field(default=2.0, gt=0)
    poisson_depth: int = Field(default=10, ge=4, le=14)
    poisson_density_quantile: float = Field(default=0.02, ge=0.0, le=1.0)
    decimation_target_triangles: int = Field(default=300_000, ge=1000)
    # Mesh (.glb) only: project vertex colours from original (unmasked) photos using estimated cameras.
    # Strongly improves realism vs point-cloud colours alone. Set false for faster jobs or if colours look wrong.
    mesh_photo_vertex_bake: bool = True
    # Write mesh GLB with Open3D compressed mode (Draco-style mesh compression for smaller web payloads).
    mesh_glb_draco_compression: bool = True
    # Public URL prefix for generated model files (must match where this API serves /output/…).
    cdn_base_url: str = "http://127.0.0.1:8000/output"
    max_images: int = Field(default=100, ge=2, le=1000)
    max_image_side: int = Field(default=1024, ge=128, le=8192)
    # When True: fake masks/points/GS PLY (demo only). When False: real checkpoints + packages required.
    allow_placeholder_pipeline: bool = False


class SettingsStore:
    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.path = self.root_dir / "config" / "runtime_settings.json"

    def load(self) -> RuntimeSettings:
        if not self.path.exists():
            settings = RuntimeSettings()
            self.save(settings)
            return settings

        data = json.loads(self.path.read_text(encoding="utf-8"))
        return RuntimeSettings.model_validate(data)

    def save(self, settings: RuntimeSettings) -> RuntimeSettings:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
        return settings

