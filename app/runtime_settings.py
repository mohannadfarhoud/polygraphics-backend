from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    # `mapanything`/`dust3r`/`colmap` mesh to `.glb`; `gaussian_splatting` yields `.ply` splats.
    reconstruction_backend: Literal["mapanything", "dust3r", "colmap", "gaussian_splatting"] = "mapanything"

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
    dust3r_checkpoint_path: str | None = None
    dust3r_aligner_iters: int = Field(default=300, ge=10, le=4000)
    dust3r_aligner_lr: float = Field(default=0.01, gt=0.0, le=1.0)
    dust3r_confidence_threshold: float = Field(default=0.1, ge=0.0, le=1.0)
    dust3r_use_fp16: bool = True
    dust3r_inference_batch_size: int = Field(default=1, ge=1, le=16)
    dust3r_max_inference_side: int = Field(default=768, ge=224, le=2048)
    dust3r_max_input_views: int = Field(default=36, ge=2, le=256)
    dust3r_scene_graph: str = "auto"
    dust3r_complete_graph_max_views: int = Field(default=24, ge=2, le=256)
    # Legacy COLMAP SfM backend (optional). Set this on the machine running reconstruction.
    colmap_binary_path: str | None = None
    colmap_sift_gpu: bool = True

    device: Literal["auto", "cpu", "cuda"] = "auto"
    # When True (default): phase 1 (SAM) and GS MapAnything scene prep run in subprocesses on CUDA so
    # each PyTorch workload exits before the next — recommended on single 8 GB GPUs.
    gpu_isolate_phases: bool = True

    # When True: skip Segment Anything; uploads must be **RGBA cutouts** with a real alpha matte
    # (transparent / feathered background). Plain RGB or fully opaque images are rejected — use
    # skip_sam_segmentation=false so SAM removes the background first on full-frame photos.
    # RGBA alpha is flattened onto ``mapanything_alpha_flatten_gray`` grey before MapAnything reads views.
    skip_sam_segmentation: bool = False
    # Phase-1 foreground isolation backend when skip_sam_segmentation=false.
    isolation_backend: Literal["sam", "rembg"] = "sam"
    # When true: evaluate multiple isolation candidates (configured backend + optional fallback backend)
    # and keep the most plausible centered object mask.
    isolation_smart_select: bool = True
    # When true: also try the non-selected backend as a fallback candidate (SAM <-> rembg).
    isolation_try_alternate_backend: bool = True
    # Minimum mask quality score accepted by smart selector; lower values are treated as unsafe/noisy masks.
    isolation_min_score: float = Field(default=1.1, ge=-10.0, le=10.0)
    # rembg model choice (e.g. "isnet-general-use", "u2net", "u2netp", "birefnet-general").
    rembg_model_name: str = "isnet-general-use"
    # rembg output alpha threshold (0-255) to convert matte into binary mask.
    rembg_alpha_threshold: int = Field(default=16, ge=0, le=255)
    # After phase-1 isolation, translate the masked subject so its foreground centroid sits at image center.
    recenter_isolated_subject: bool = True
    # Target relative size for isolated subject before saving masked images (based on max bbox side / image min side).
    recenter_target_subject_fill: float = Field(default=0.62, gt=0.05, le=0.98)
    # Limit enlargement factor during recentering (1.0 = no scaling, >1 enlarges object in frame).
    recenter_max_scale: float = Field(default=1.85, ge=1.0, le=4.0)
    # Reject clearly inconsistent segmented views before reconstruction (keeps most coherent masked set).
    isolation_filter_outlier_views: bool = True
    # How strict cross-view mask-area filtering is (higher keeps more views).
    isolation_outlier_area_mad_scale: float = Field(default=3.2, ge=1.0, le=8.0)
    # Max normalized centroid drift from median foreground center before dropping a view.
    isolation_outlier_center_distance: float = Field(default=0.22, ge=0.05, le=0.7)

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
    # Prompt-based SAM sanity bounds; helps avoid selecting a "whole background" mask when the centred subject is tiny.
    sam_prompt_min_mask_area_ratio: float = Field(default=0.0005, ge=0.0, le=0.2)
    sam_prompt_max_mask_area_ratio: float = Field(default=0.3, ge=0.05, le=0.98)
    # When true: prompt-based selection only accepts masks that still contain the image centre.
    # If no such candidate exists, auto center-bias recovery is attempted (when enabled below).
    sam_prompt_require_center_hit: bool = True
    # If a prompt-based mask falls outside area bounds, retry with auto center-biased selection and prefer it when tighter.
    sam_recover_with_auto_if_prompt_bad: bool = True

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
    # When baking: sample ``masked`` (black outside SAM FG) vs full ``original`` photos.
    # Default to ``original`` for stronger texture realism on the object surface.
    mesh_photo_vertex_bake_sample_source: Literal["masked", "original"] = "original"
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
        if rb == "auto":
            d["reconstruction_backend"] = "mapanything"

        preview = bool(d.get("compare_mesh_preview_with_gs", False)) or bool(
            d.pop("compare_mesh_dust3r_colmap_with_gs", False)
        )
        d["compare_mesh_preview_with_gs"] = preview

        # Drop obsolete keys silently (were ignored via extra="ignore" but tidy common ones).
        for dead in (
            "mesh_colmap_failure_fallback_dust3r",
            "auto_dust3r_max_images",
            "gs_init_source",
        ):
            d.pop(dead, None)
        return d


def gaussian_splatting_skipped_via_env() -> bool:
    return os.getenv("POLYGRAPH_SKIP_GAUSSIAN_SPLATTING", "").strip().lower() in ("1", "true", "yes")


def effective_reconstruction_backend(
    settings: RuntimeSettings,
) -> Literal["mapanything", "dust3r", "colmap", "gaussian_splatting"]:
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
