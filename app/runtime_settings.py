from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    # `mapanything` meshes a `.glb`; `gaussian_splatting` yields a `.ply` (3DGS, seeded from MapAnything → COLMAP-text).
    reconstruction_backend: Literal["mapanything", "gaussian_splatting"] = "mapanything"

    # Hugging Face model id (or local snapshot path) for Meta MapAnything.
    mapanything_pretrained_id: str = "facebook/map-anything-apache"
    # When ``skip_sam_segmentation`` and inputs have alpha: ``alpha=0`` RGB channels become this grey (0–255 each).
    mapanything_alpha_flatten_gray: int = Field(default=200, ge=0, le=255)
    mapanything_memory_efficient_inference: bool = True
    mapanything_minibatch_size: int = Field(default=1, ge=1, le=128)
    mapanything_use_amp: bool = True
    mapanything_apply_mask: bool = True
    mapanything_mask_edges: bool = True
    mapanything_apply_confidence_mask: bool = False
    mapanything_confidence_percentile: int = Field(default=10, ge=0, le=99)
    mapanything_use_multiview_confidence: bool = False
    mapanything_max_input_views: int = Field(default=48, ge=2, le=512)
    # When ``skip_sam_segmentation``: subject is already cut out; MapAnything's internal inference
    # mask often erodes specular / toy paint — keep false unless foreground is fused with noisy BG.
    mapanything_apply_internal_mask_on_precut: bool = False

    device: Literal["auto", "cpu", "cuda"] = "auto"
    # When True (default): phase 1 (SAM) and GS MapAnything scene prep run in subprocesses on CUDA so
    # each PyTorch workload exits before the next — recommended on single 8 GB GPUs.
    gpu_isolate_phases: bool = True

    # When True: skip Segment Anything; uploads must be **RGBA cutouts** with a real alpha matte
    # (transparent / feathered background). Plain RGB or fully opaque images are rejected — use
    # skip_sam_segmentation=false so SAM removes the background first on full-frame photos.
    # RGBA alpha is flattened onto ``mapanything_alpha_flatten_gray`` grey before MapAnything reads views.
    skip_sam_segmentation: bool = False

    sam_checkpoint_path: str | None = None
    # Must match the .pth file: ``sam_vit_b_*.pth`` → vit_b; ``sam_vit_h_*.pth`` → vit_h; ``sam_vit_l_*.pth`` → vit_l.
    sam_model_type: Literal["vit_h", "vit_l", "vit_b"] = "vit_b"
    sam_segmentation_mode: Literal[
        "center_subject",
        "center_subject_table",
        "center_point",
        "auto_masks_center_bias",
        "auto_masks_largest_area",
    ] = "center_subject_table"
    sam_use_fp16: bool = True
    # For ``center_subject_table``: number of extra background clicks along the bottom edge (table plane). 0 = corners only.
    sam_table_edge_negative_points: int = Field(default=11, ge=0, le=24)

    # When True: copy SAM/precut isolated views (``masked_*``) under ``uploads/{job_id}/masked_views/`` for HTTP GET.
    # Remote GPU workers POST the same files to the API after a successful run.
    expose_masked_views: bool = False

    # With ``reconstruction_backend == "gaussian_splatting"``, emit ``<job_id>_compare_mesh.glb`` (MapAnything mesh)
    # before neural GS training when True.
    compare_mesh_preview_with_gs: bool = False

    gs_repo_path: str | None = None
    gs_python_executable: str | None = None  # leave null to use the API's Python
    gs_iterations: int = Field(default=10000, ge=100, le=60000)
    gs_sh_degree: int = Field(default=2, ge=0, le=4)
    gs_densify_until_iter: int = Field(default=7000, ge=0, le=60000)
    gs_resolution: int = Field(default=-1, ge=-1, le=8192)  # -1 = original
    gs_opacity_reset_interval: int = Field(default=3000, ge=100, le=60000)
    gs_allow_cpu_fallback: bool = True
    gs_cpu_max_points: int = Field(default=250_000, ge=1000, le=2_000_000)
    gs_train_with_original_images: bool = True

    meshing_method: Literal["poisson", "bpa"] = "poisson"
    output_dir_name: str = "output"
    masked_dir_name: str = "masked"
    masks_dir_name: str = "masks"
    save_raw_masks: bool = True
    nb_neighbors: int = Field(default=26, ge=1)
    std_ratio: float = Field(default=2.0, gt=0)
    poisson_depth: int = Field(default=10, ge=4, le=14)
    poisson_density_quantile: float = Field(default=0.02, ge=0.0, le=1.0)
    decimation_target_triangles: int = Field(default=520_000, ge=1000)
    mesh_photo_vertex_bake: bool = True
    # When baking: sample ``masked`` (black outside SAM FG) vs full ``original`` photos — originals can smear backdrop onto the mesh.
    mesh_photo_vertex_bake_sample_source: Literal["masked", "original"] = "masked"
    mesh_glb_draco_compression: bool = True
    cdn_base_url: str = "http://127.0.0.1:8000/output"
    max_images: int = Field(default=100, ge=2, le=1000)
    max_input_image_side: int = Field(default=1920, ge=256, le=8192)
    max_image_side: int = Field(default=1024, ge=128, le=8192)
    allow_placeholder_pipeline: bool = False

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        d = dict(data)
        rb = d.get("reconstruction_backend")
        if rb in ("auto", "dust3r", "colmap"):
            d["reconstruction_backend"] = "mapanything"

        preview = bool(d.get("compare_mesh_preview_with_gs", False)) or bool(
            d.pop("compare_mesh_dust3r_colmap_with_gs", False)
        )
        d["compare_mesh_preview_with_gs"] = preview

        # Drop obsolete keys silently (were ignored via extra="ignore" but tidy common ones).
        for dead in (
            "mesh_colmap_failure_fallback_dust3r",
            "auto_dust3r_max_images",
            "dust3r_repo_path",
            "dust3r_checkpoint_path",
            "dust3r_aligner_iters",
            "dust3r_aligner_lr",
            "dust3r_confidence_threshold",
            "dust3r_use_fp16",
            "dust3r_inference_batch_size",
            "dust3r_max_inference_side",
            "dust3r_max_input_views",
            "dust3r_scene_graph",
            "dust3r_complete_graph_max_views",
            "colmap_binary_path",
            "colmap_sift_gpu",
            "gs_init_source",
        ):
            d.pop(dead, None)
        return d


def gaussian_splatting_skipped_via_env() -> bool:
    return os.getenv("POLYGRAPH_SKIP_GAUSSIAN_SPLATTING", "").strip().lower() in ("1", "true", "yes")


def effective_reconstruction_backend(
    settings: RuntimeSettings,
) -> Literal["mapanything", "gaussian_splatting"]:
    """What the pipeline actually runs (`POLYGRAPH_SKIP_GAUSSIAN_SPLATTING` forces mesh path)."""
    if gaussian_splatting_skipped_via_env():
        return "mapanything"
    return settings.reconstruction_backend


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
