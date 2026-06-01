from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    # `auto` selects backend by image count automatically (recommended default).
    # `mapanything`/`dust3r`/`colmap` mesh to `.glb`; `gaussian_splatting` yields `.ply` splats.
    reconstruction_backend: Literal[
        "auto", "mapanything", "dust3r", "colmap", "gaussian_splatting", "ai_prior", "hybrid_prior_refine"
    ] = "auto"

    # ---------------------------------------------------------------------------
    # Auto-selection thresholds (used when reconstruction_backend == "auto").
    # ---------------------------------------------------------------------------
    # Use TripoSR (single-image AI) when image count is at most this value.
    # Set 0 to never auto-select TripoSR.
    auto_backend_triposr_max_images: int = Field(default=3, ge=0, le=100)
    # Use DUSt3R when image count is above triposr threshold and at most this value.
    # Set 0 to skip DUSt3R and go straight to MapAnything.
    auto_backend_dust3r_max_images: int = Field(default=15, ge=0, le=256)
    # Image count above dust3r threshold always routes to MapAnything.
    # Which images feed geometry matching/reconstruction. `original` is more robust for sparse matching;
    # `masked` keeps strict object-only context but can reduce feature richness on low-texture objects.
    reconstruction_image_source: Literal["original", "masked"] = "original"
    # Capture quality gate before segmentation/reconstruction.
    capture_quality_gate_enabled: bool = True
    capture_reject_policy: Literal["soft", "hard"] = "soft"
    capture_min_kept_images: int = Field(default=8, ge=2, le=200)
    capture_blur_min: float = Field(default=35.0, ge=0.0, le=10000.0)
    capture_brightness_min: float = Field(default=20.0, ge=0.0, le=255.0)
    capture_brightness_max: float = Field(default=235.0, ge=0.0, le=255.0)
    capture_min_frame_delta: float = Field(default=0.010, ge=0.0, le=1.0)
    capture_max_selected_images: int = Field(default=20, ge=2, le=256)
    capture_duplicate_similarity: float = Field(default=0.995, ge=0.8, le=1.0)
    capture_diversity_min_distance: float = Field(default=0.045, ge=0.0, le=1.0)
    # Post-mask depth consistency normalization before reconstruction (proxy depth from object footprint).
    depth_normalization_enabled: bool = True
    depth_reference_mode: Literal["median_best", "first"] = "median_best"
    depth_consistency_apply_filter: bool = True
    depth_consistency_max_relative: float = Field(default=0.35, ge=0.05, le=2.0)
    depth_consistency_min_kept_images: int = Field(default=3, ge=2, le=256)
    depth_consistency_min_score: float = Field(default=0.40, ge=0.0, le=1.0)
    depth_consistency_fail_on_low_score: bool = False
    depth_selection_quality_weight: float = Field(default=0.55, ge=0.0, le=1.0)
    depth_selection_proximity_weight: float = Field(default=0.45, ge=0.0, le=1.0)
    # Confidence routing across AI-prior and hybrid pipelines.
    reconstruction_confidence_high_threshold: float = Field(default=0.72, ge=0.0, le=1.0)
    reconstruction_confidence_min_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    reconstruction_low_confidence_policy: Literal["fail", "prior_only", "coarse_prior"] = "prior_only"
    # ---------------------------------------------------------------------------
    # Video input mode — accept a single video instead of image collection.
    # ---------------------------------------------------------------------------
    video_input_enabled: bool = False
    # Frames-per-second to extract from the uploaded video (2 FPS ≈ 30 frames for 15s video).
    video_input_extraction_fps: float = Field(default=2.0, ge=0.1, le=30.0)
    # Hard cap on extracted frames sent to quality scoring (subsampled uniformly if exceeded).
    video_input_max_frames: int = Field(default=60, ge=5, le=300)
    # Reject input videos longer than this (seconds). Set 0 to disable.
    video_input_max_duration_seconds: int = Field(default=30, ge=0, le=600)
    # Maximum upload size for a video file in megabytes. 0 = no limit (rely on OS).
    video_input_max_file_mb: int = Field(default=500, ge=0, le=10000)
    # Frame filter thresholds (frames scoring below these are discarded before best-frame selection).
    video_frame_min_sharpness: float = Field(default=0.04, ge=0.0, le=1.0)
    video_frame_min_exposure: float = Field(default=0.10, ge=0.0, le=1.0)
    # FFmpeg binary path; leave empty to use system PATH.
    video_ffmpeg_binary: str | None = None

    # ---------------------------------------------------------------------------
    # AI-prior backend options (pluggable provider adapter).
    # ---------------------------------------------------------------------------
    ai_prior_provider: Literal["command", "mock", "triposr_local", "instantmesh_local"] = "command"
    ai_prior_command: str | None = None
    ai_prior_command_args_template: str = "--input-manifest {input_manifest} --output {output_mesh}"
    ai_prior_output_mesh_path: str | None = None
    ai_prior_timeout_seconds: int = Field(default=600, ge=30, le=7200)
    ai_prior_api_key_env: str = "AI_PRIOR_API_KEY"
    ai_prior_require_api_key: bool = False
    ai_prior_default_confidence: float = Field(default=0.62, ge=0.0, le=1.0)
    # When true: bypass confidence routing and keep prior-only route (useful for simple TripoSR-local flow).
    ai_prior_force_prior_only: bool = False
    # TripoSR local provider options (no cloud API credits; runs model on worker machine).
    ai_prior_triposr_repo_path: str | None = None
    ai_prior_triposr_python_executable: str | None = None
    ai_prior_triposr_entry_script: str = "run.py"
    ai_prior_triposr_args_template: str = "{input_image} --output-dir {output_dir}"
    # InstantMesh local provider (Tencent InstantMesh single-image-to-3D).
    ai_prior_instantmesh_repo_path: str | None = None
    ai_prior_instantmesh_python_executable: str | None = None
    ai_prior_instantmesh_entry_script: str = "run.py"
    ai_prior_instantmesh_args_template: str = "--input {input_image} --output-dir {output_dir}"
    # InstantMesh post-processing options.
    instantmesh_post_process: bool = True
    instantmesh_decimate_target: int | None = None
    # Hybrid prior-refinement route settings.
    hybrid_refine_backend: Literal["mapanything", "dust3r", "colmap", "none"] = "mapanything"
    hybrid_refine_strength: float = Field(default=0.20, ge=0.0, le=1.0)
    hybrid_min_refine_points: int = Field(default=5000, ge=100, le=5_000_000)
    hybrid_enable_photo_bake: bool = True
    # Optional semantic shape prior stage (classify + template-based correction).
    shape_prior_enabled: bool = False
    # Template directory containing class meshes (example: templates/car.glb, templates/truck.obj).
    shape_prior_template_root: str | None = None
    # Minimum classification confidence to apply template correction.
    shape_prior_min_confidence: float = Field(default=0.34, ge=0.0, le=1.0)
    # Fallback class label when classifier is unavailable/uncertain and only one class is desired.
    shape_prior_force_label: str | None = None
    # Amount of deformation toward template surface (0 keeps original mesh, 1 snaps to template).
    shape_prior_deform_strength: float = Field(default=0.22, ge=0.0, le=1.0)
    # Keep this fraction of original geometry detail (higher preserves more of the original mesh).
    shape_prior_preserve_detail: float = Field(default=0.78, ge=0.0, le=1.0)
    # Disable template correction when alignment RMSE exceeds this threshold (normalized units).
    shape_prior_max_alignment_rmse: float = Field(default=0.12, ge=0.001, le=1.0)

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
    isolation_min_score: float = Field(default=0.95, ge=-10.0, le=10.0)
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
    # If false (recommended): recenter performs translation only and keeps per-view scale unchanged.
    recenter_allow_per_image_scaling: bool = False
    # Reject clearly inconsistent segmented views before reconstruction (keeps most coherent masked set).
    isolation_filter_outlier_views: bool = True
    # How strict cross-view mask-area filtering is (higher keeps more views).
    isolation_outlier_area_mad_scale: float = Field(default=3.2, ge=1.0, le=8.0)
    # Max normalized centroid drift from median foreground center before dropping a view.
    isolation_outlier_center_distance: float = Field(default=0.22, ge=0.05, le=0.7)
    # If true: fail the job when no plausible mask is found instead of silently using ellipse fallback.
    segmentation_fail_fast: bool = True

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
    # Optional surface abstraction before photo vertex bake (flattens harsh lighting/shadows).
    texture_surface_abstraction_enabled: bool = False
    texture_surface_abstraction_strength: float = Field(default=0.42, ge=0.0, le=1.0)
    texture_surface_detail_preserve: float = Field(default=0.70, ge=0.0, le=1.0)
    texture_surface_illumination_blur: int = Field(default=41, ge=9, le=301)
    # Optional dominant-surface extraction per texture source image before projection.
    surface_region_texture_enabled: bool = False
    surface_region_smooth_percentile: float = Field(default=85.0, ge=50.0, le=98.0)
    surface_region_min_area_ratio: float = Field(default=0.08, ge=0.02, le=0.25)
    surface_region_expand_px: int = Field(default=8, ge=0, le=32)
    region_projection_blend_weight: float = Field(default=0.65, ge=0.2, le=0.9)
    region_projection_min_confidence: float = Field(default=0.60, ge=0.3, le=0.9)
    region_projection_seam_smoothing: float = Field(default=0.55, ge=0.0, le=1.0)
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

        preview = bool(d.get("compare_mesh_preview_with_gs", False)) or bool(
            d.pop("compare_mesh_dust3r_colmap_with_gs", False)
        )
        d["compare_mesh_preview_with_gs"] = preview
        provider = str(d.get("ai_prior_provider", "")).strip().lower()
        if provider == "triposr":
            d["ai_prior_provider"] = "triposr_local"
        high = d.get("reconstruction_confidence_high_threshold")
        low = d.get("reconstruction_confidence_min_threshold")
        try:
            if high is not None and low is not None and float(high) < float(low):
                d["reconstruction_confidence_high_threshold"] = float(low)
        except Exception:
            pass
        try:
            qw = float(d.get("depth_selection_quality_weight", 0.55))
            pw = float(d.get("depth_selection_proximity_weight", 0.45))
            if qw <= 0.0 and pw <= 0.0:
                d["depth_selection_quality_weight"] = 0.55
                d["depth_selection_proximity_weight"] = 0.45
        except Exception:
            pass

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
) -> Literal["auto", "mapanything", "dust3r", "colmap", "gaussian_splatting", "ai_prior", "hybrid_prior_refine"]:
    """What the pipeline actually runs (`POLYGRAPH_SKIP_GAUSSIAN_SPLATTING` forces mesh path).

    Returns ``"auto"`` unchanged — the pipeline resolves it to a concrete backend after
    the image count is known. Use ``resolve_auto_backend`` for a concrete value.
    """
    if gaussian_splatting_skipped_via_env():
        return "mapanything"
    return settings.reconstruction_backend


def resolve_auto_backend(
    settings: RuntimeSettings,
    image_count: int,
) -> tuple[str, str | None]:
    """Resolve ``"auto"`` backend to a concrete backend + optional ai_prior_provider.

    Selection logic (thresholds configurable via settings):

    * 1 .. auto_backend_triposr_max_images  → triposr_local (when repo is configured)
    * (triposr_max+1) .. auto_backend_dust3r_max_images → dust3r
    * above dust3r_max → mapanything

    Returns (backend, ai_provider_override_or_none).
    """
    triposr_max = int(getattr(settings, "auto_backend_triposr_max_images", 3))
    dust3r_max = int(getattr(settings, "auto_backend_dust3r_max_images", 15))

    # TripoSR path: single-image AI — only use if repo is configured on worker.
    triposr_repo = str(getattr(settings, "ai_prior_triposr_repo_path", "") or "").strip()
    triposr_available = bool(triposr_repo)

    if triposr_max > 0 and image_count <= triposr_max and triposr_available:
        return "ai_prior", "triposr_local"

    # DUSt3R path: better geometry for few images.
    if dust3r_max > 0 and image_count <= dust3r_max:
        return "dust3r", None

    # MapAnything: fast feed-forward for many images.
    return "mapanything", None


def minimum_input_images(settings: RuntimeSettings) -> int:
    """Minimum input count expected by the active backend path."""
    backend = str(effective_reconstruction_backend(settings)).strip().lower()
    # auto can handle a single image (TripoSR branch).
    if backend == "auto":
        triposr_max = int(getattr(settings, "auto_backend_triposr_max_images", 3))
        if triposr_max > 0:
            return 1
        return 2
    if backend == "ai_prior":
        provider = str(getattr(settings, "ai_prior_provider", "")).strip().lower()
        if provider in ("triposr_local", "instantmesh_local"):
            # Both are single-image models; also accept video upload (1 video → frames).
            return 1
        if provider == "command" and bool(getattr(settings, "ai_prior_force_prior_only", False)):
            # Command-mode single-image generators (e.g. Tripo API) can run with one view
            # when we intentionally keep a strict prior-only route.
            return 1
    return 2


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
