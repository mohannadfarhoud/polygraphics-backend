"""Human-oriented metadata for ``RuntimeSettings`` fields (API vs worker vs both)."""

from __future__ import annotations

from typing import Any


def settings_deployment_guide() -> dict[str, Any]:
    """Return JSON for ``GET /settings/deployment`` — what applies on API host vs GPU worker."""
    fields: list[dict[str, Any]] = [
        {
            "key": "reconstruction_backend",
            "scope": "both",
            "worker_env": None,
            "notes": (
                "`auto` (recommended default): selects backend by image count automatically — "
                "TripoSR for 1–auto_backend_triposr_max_images images, DUSt3R for a few more, MapAnything for many. "
                "Manual options: mapanything (.glb), dust3r (.glb), colmap (.glb), ai_prior (.glb), "
                "hybrid_prior_refine (.glb); gaussian_splatting (.ply) for splats."
            ),
        },
        {
            "key": "auto_backend_triposr_max_images",
            "scope": "both",
            "worker_env": None,
            "notes": (
                "Only when reconstruction_backend=auto: use TripoSR (single-image AI) when "
                "image count is at most this value (default 3). One image always routes to "
                "TripoSR/InstantMesh regardless of this threshold. Set 0 to skip TripoSR for "
                "2+ image jobs (those fall through to DUSt3R/MapAnything)."
            ),
        },
        {
            "key": "auto_backend_dust3r_max_images",
            "scope": "both",
            "worker_env": None,
            "notes": (
                "Only when reconstruction_backend=auto: use DUSt3R when image count is above "
                "auto_backend_triposr_max_images and at most this value. Default 15. "
                "Above this threshold MapAnything is used. Set 0 to skip DUSt3R."
            ),
        },
        {
            "key": "reconstruction_image_source",
            "scope": "both",
            "worker_env": None,
            "notes": "`original` (recommended): use full photos for feature matching/geometry, while masks are still used for isolation + cleanup. `masked`: use black-background isolated views directly for reconstruction.",
        },
        {
            "key": "capture_quality_gate_enabled",
            "scope": "both",
            "worker_env": None,
            "notes": "Enable pre-segmentation capture quality filtering (blur/exposure/camera-motion consistency) and write uploads/{job_id}/capture_quality_report.json.",
        },
        {
            "key": "capture_reject_policy",
            "scope": "both",
            "worker_env": None,
            "notes": "`soft`: keep all images if too many would be dropped. `hard`: fail the job when kept images are below capture_min_kept_images.",
        },
        {
            "key": "capture_min_kept_images",
            "scope": "both",
            "worker_env": None,
            "notes": "Minimum images required after capture quality filtering.",
        },
        {
            "key": "capture_blur_min",
            "scope": "both",
            "worker_env": None,
            "notes": "Minimum Laplacian variance blur score for a frame to be considered sharp enough.",
        },
        {
            "key": "capture_brightness_min",
            "scope": "both",
            "worker_env": None,
            "notes": "Minimum mean grayscale brightness allowed by the capture quality gate.",
        },
        {
            "key": "capture_brightness_max",
            "scope": "both",
            "worker_env": None,
            "notes": "Maximum mean grayscale brightness allowed by the capture quality gate.",
        },
        {
            "key": "capture_min_frame_delta",
            "scope": "both",
            "worker_env": None,
            "notes": "Minimum inter-frame pixel delta; very low values are treated as duplicate/low-motion captures.",
        },
        {
            "key": "capture_max_selected_images",
            "scope": "both",
            "worker_env": None,
            "notes": "Maximum number of curated images kept for downstream reconstruction after filtering and diversity selection.",
        },
        {
            "key": "capture_duplicate_similarity",
            "scope": "both",
            "worker_env": None,
            "notes": "Cosine-similarity threshold for duplicate suppression in capture curation (higher = stricter duplicate rejection).",
        },
        {
            "key": "capture_diversity_min_distance",
            "scope": "both",
            "worker_env": None,
            "notes": "Minimum tiny-image distance between curated frames (higher encourages more diverse viewing angles).",
        },
        {
            "key": "depth_normalization_enabled",
            "scope": "both",
            "worker_env": None,
            "notes": "Run post-mask depth normalization before reconstruction (select robust reference view and score consistency).",
        },
        {
            "key": "depth_reference_mode",
            "scope": "both",
            "worker_env": None,
            "notes": "Reference strategy for depth normalization: `median_best` (recommended) or `first`.",
        },
        {
            "key": "depth_consistency_apply_filter",
            "scope": "both",
            "worker_env": None,
            "notes": "When true, drop views with depth proxy deviation above threshold after masking.",
        },
        {
            "key": "depth_consistency_max_relative",
            "scope": "both",
            "worker_env": None,
            "notes": "Maximum relative depth proxy deviation from reference before a view is rejected.",
        },
        {
            "key": "depth_consistency_min_kept_images",
            "scope": "both",
            "worker_env": None,
            "notes": "Minimum views to keep after depth filtering; falls back to keep-all when stricter filtering would go below this.",
        },
        {
            "key": "depth_consistency_min_score",
            "scope": "both",
            "worker_env": None,
            "notes": "Minimum acceptable depth consistency score (0..1). Used only when depth_consistency_fail_on_low_score is true.",
        },
        {
            "key": "depth_consistency_fail_on_low_score",
            "scope": "both",
            "worker_env": None,
            "notes": "When true, fail job early if depth consistency score is below depth_consistency_min_score.",
        },
        {
            "key": "depth_selection_quality_weight",
            "scope": "both",
            "worker_env": None,
            "notes": "Weight of image quality (blur/exposure) in selecting the preferred Tripo input frame.",
        },
        {
            "key": "depth_selection_proximity_weight",
            "scope": "both",
            "worker_env": None,
            "notes": "Weight of depth proximity-to-reference in selecting the preferred Tripo input frame.",
        },
        {
            "key": "reconstruction_confidence_high_threshold",
            "scope": "both",
            "worker_env": None,
            "notes": "Confidence threshold to route jobs into hybrid prior-refinement path.",
        },
        {
            "key": "reconstruction_confidence_min_threshold",
            "scope": "both",
            "worker_env": None,
            "notes": "Minimum confidence threshold to allow prior-only route instead of failing for weak captures.",
        },
        {
            "key": "reconstruction_low_confidence_policy",
            "scope": "both",
            "worker_env": None,
            "notes": "Routing policy when confidence falls below reconstruction_confidence_min_threshold (`fail`, `prior_only`, or `coarse_prior`).",
        },
        {
            "key": "video_input_enabled",
            "scope": "both",
            "worker_env": None,
            "notes": "When true: pipeline accepts a single uploaded video, extracts frames with FFmpeg, filters by quality, and feeds the best frame to reconstruction. Use POST /jobs/video or POST /jobs/video/reconstruct.",
        },
        {
            "key": "video_input_extraction_fps",
            "scope": "both",
            "worker_env": None,
            "notes": "Frames per second to extract from uploaded video (default 2.0 ≈ 30 frames for a 15s clip).",
        },
        {
            "key": "video_input_max_frames",
            "scope": "both",
            "worker_env": None,
            "notes": "Maximum extracted frames passed to quality scoring (subsampled uniformly when exceeded).",
        },
        {
            "key": "video_input_max_duration_seconds",
            "scope": "both",
            "worker_env": None,
            "notes": "Advisory maximum video duration in seconds (0 = no limit). Logs a warning when exceeded.",
        },
        {
            "key": "video_input_max_file_mb",
            "scope": "both",
            "worker_env": None,
            "notes": "Maximum video upload size in megabytes. 0 = no limit (rely on OS/nginx limits).",
        },
        {
            "key": "video_frame_min_sharpness",
            "scope": "both",
            "worker_env": None,
            "notes": "Frames with normalised sharpness below this threshold are discarded before best-frame selection.",
        },
        {
            "key": "video_frame_min_exposure",
            "scope": "both",
            "worker_env": None,
            "notes": "Frames with normalised exposure below this threshold (too dark or overexposed) are discarded.",
        },
        {
            "key": "video_ffmpeg_binary",
            "scope": "stored_on_api",
            "worker_env": "POLYGRAPH_OVERRIDE_FFMPEG_BINARY (optional path override)",
            "notes": "Path to the FFmpeg executable. Leave null to use the system PATH.",
        },
        {
            "key": "ai_prior_provider",
            "scope": "both",
            "worker_env": None,
            "notes": "AI prior provider adapter (`command`, `triposr_local`, `instantmesh_local`, or `mock`). `command` runs external executable/script; `triposr_local` runs local TripoSR; `instantmesh_local` runs local InstantMesh.",
        },
        {
            "key": "ai_prior_command",
            "scope": "stored_on_api",
            "worker_env": "POLYGRAPH_OVERRIDE_AI_PRIOR_COMMAND (optional via wrapper env export)",
            "notes": "Executable/script path (or command on PATH) used when ai_prior_provider=command.",
        },
        {
            "key": "ai_prior_command_args_template",
            "scope": "both",
            "worker_env": None,
            "notes": "Argument template for ai_prior_command. Tokens: {job_id}, {input_manifest}, {output_mesh}, {output_dir}.",
        },
        {
            "key": "ai_prior_output_mesh_path",
            "scope": "both",
            "worker_env": None,
            "notes": "Optional explicit output mesh path if provider ignores {output_mesh}.",
        },
        {
            "key": "ai_prior_timeout_seconds",
            "scope": "both",
            "worker_env": None,
            "notes": "Timeout for external AI-prior provider command execution.",
        },
        {
            "key": "ai_prior_api_key_env",
            "scope": "both",
            "worker_env": "AI_PRIOR_API_KEY",
            "notes": "Environment variable name that stores provider API key (passed through to command provider).",
        },
        {
            "key": "ai_prior_require_api_key",
            "scope": "both",
            "worker_env": None,
            "notes": "When true, fail fast unless env var named by ai_prior_api_key_env is set on the machine running the job.",
        },
        {
            "key": "ai_prior_default_confidence",
            "scope": "both",
            "worker_env": None,
            "notes": "Default prior-model confidence value used in routing/reporting when provider does not emit confidence.",
        },
        {
            "key": "ai_prior_force_prior_only",
            "scope": "both",
            "worker_env": None,
            "notes": "When true: skip confidence-based fail/hybrid routing and keep prior-only route (simple Tripo-first flow). Also enables 1-image minimum when ai_prior_provider=command (for single-image cloud generators like Tripo API).",
        },
        {
            "key": "ai_prior_triposr_repo_path",
            "scope": "stored_on_api",
            "worker_env": "POLYGRAPH_OVERRIDE_TRIPOSR_REPO",
            "notes": (
                "Required when ai_prior_provider=triposr_local: local path to TripoSR repository on the worker. "
                "TripoSR mode skips SAM/background removal — the original photo is passed directly to TripoSR."
            ),
        },
        {
            "key": "ai_prior_triposr_python_executable",
            "scope": "stored_on_api",
            "worker_env": "POLYGRAPH_OVERRIDE_TRIPOSR_PYTHON",
            "notes": "Optional Python executable for TripoSR local provider. Defaults to current worker Python.",
        },
        {
            "key": "ai_prior_triposr_entry_script",
            "scope": "both",
            "worker_env": None,
            "notes": "Entry script relative to ai_prior_triposr_repo_path (default run.py).",
        },
        {
            "key": "ai_prior_triposr_args_template",
            "scope": "both",
            "worker_env": None,
            "notes": "CLI argument template for TripoSR local provider. Tokens: {job_id}, {input_image}, {output_dir}, {output_mesh}, {repo_path}.",
        },
        {
            "key": "ai_prior_instantmesh_repo_path",
            "scope": "stored_on_api",
            "worker_env": "POLYGRAPH_OVERRIDE_INSTANTMESH_REPO",
            "notes": "Required when ai_prior_provider=instantmesh_local: local path to InstantMesh repository on the worker.",
        },
        {
            "key": "ai_prior_instantmesh_python_executable",
            "scope": "stored_on_api",
            "worker_env": "POLYGRAPH_OVERRIDE_INSTANTMESH_PYTHON",
            "notes": "Optional Python executable for InstantMesh local provider. Defaults to current worker Python.",
        },
        {
            "key": "ai_prior_instantmesh_entry_script",
            "scope": "both",
            "worker_env": None,
            "notes": "Entry script relative to ai_prior_instantmesh_repo_path (default run.py).",
        },
        {
            "key": "ai_prior_instantmesh_args_template",
            "scope": "both",
            "worker_env": None,
            "notes": "CLI argument template for InstantMesh local provider. Tokens: {job_id}, {input_image}, {output_dir}, {repo_path}.",
        },
        {
            "key": "instantmesh_post_process",
            "scope": "both",
            "worker_env": None,
            "notes": "When true (default): remove floating artifacts and re-export mesh after InstantMesh generation.",
        },
        {
            "key": "instantmesh_decimate_target",
            "scope": "both",
            "worker_env": None,
            "notes": "Optional triangle target for decimation during InstantMesh post-processing. Null = no decimation.",
        },
        {
            "key": "hybrid_refine_backend",
            "scope": "both",
            "worker_env": None,
            "notes": "Backend used for refinement signal in hybrid_prior_refine route (`mapanything`, `dust3r`, `colmap`, or `none`).",
        },
        {
            "key": "hybrid_refine_strength",
            "scope": "both",
            "worker_env": None,
            "notes": "Blend strength when nudging AI-prior mesh vertices toward reconstructed point cloud.",
        },
        {
            "key": "hybrid_min_refine_points",
            "scope": "both",
            "worker_env": None,
            "notes": "Minimum aligned points required before applying geometric hybrid refinement.",
        },
        {
            "key": "hybrid_enable_photo_bake",
            "scope": "both",
            "worker_env": None,
            "notes": "When true in hybrid route, apply photo-vertex bake using reconstruction camera views after prior refinement.",
        },
        {
            "key": "shape_prior_enabled",
            "scope": "both",
            "worker_env": None,
            "notes": "When true: run semantic shape-prior refinement after meshing (classify object from views, align class template, apply gentle geometric correction).",
        },
        {
            "key": "shape_prior_template_root",
            "scope": "both",
            "worker_env": None,
            "notes": "Directory containing template meshes named by class label (for example car.glb, truck.obj). Used only when shape_prior_enabled=true.",
        },
        {
            "key": "shape_prior_min_confidence",
            "scope": "both",
            "worker_env": None,
            "notes": "Minimum class confidence needed before applying template correction. Lower values apply correction more often but increase wrong-class risk.",
        },
        {
            "key": "shape_prior_force_label",
            "scope": "both",
            "worker_env": None,
            "notes": "Optional fixed class label override (e.g. `car`) when classifier is unavailable or unstable.",
        },
        {
            "key": "shape_prior_deform_strength",
            "scope": "both",
            "worker_env": None,
            "notes": "How strongly mesh vertices move toward the aligned class template (0=no change, 1=strong correction).",
        },
        {
            "key": "shape_prior_preserve_detail",
            "scope": "both",
            "worker_env": None,
            "notes": "Bias toward preserving original mesh details while applying template correction. Higher values preserve more source geometry.",
        },
        {
            "key": "shape_prior_max_alignment_rmse",
            "scope": "both",
            "worker_env": None,
            "notes": "Safety gate: if template-to-mesh ICP alignment error exceeds this value, correction is skipped.",
        },
        {
            "key": "mapanything_pretrained_id",
            "scope": "both",
            "worker_env": "POLYGRAPH_OVERRIDE_MAPANYTHING_MODEL (optional Hugging Face id or path)",
            "notes": "Default facebook/map-anything-apache — Apache-licensed MapAnything weights.",
        },
        {
            "key": "mapanything_memory_efficient_inference",
            "scope": "both",
            "worker_env": None,
            "notes": "Passes through to model.infer(memory_efficient_inference=...). Recommended true on 8 GB GPUs.",
        },
        {
            "key": "mapanything_minibatch_size",
            "scope": "both",
            "worker_env": None,
            "notes": "Infer minibatch in memory-efficient mode; 1 minimizes VRAM.",
        },
        {
            "key": "mapanything_max_input_views",
            "scope": "both",
            "worker_env": None,
            "notes": "Uniformly subsample masked views before MapAnything when the job has more photos.",
        },
        {
            "key": "dust3r_checkpoint_path",
            "scope": "stored_on_api",
            "worker_env": "POLYGRAPH_OVERRIDE_DUST3R_CHECKPOINT",
            "notes": "Required when reconstruction_backend=dust3r: local .pth checkpoint path on the machine running reconstruction.",
        },
        {
            "key": "dust3r_max_input_views",
            "scope": "both",
            "worker_env": None,
            "notes": "Maximum views fed to DUSt3R (uniform subsample). Lower this on 8 GB GPUs to reduce VRAM pressure.",
        },
        {
            "key": "dust3r_scene_graph",
            "scope": "both",
            "worker_env": None,
            "notes": "DUSt3R pair graph; keep `auto` to use complete graph for small sets and sliding window for large sets.",
        },
        {
            "key": "dust3r_confidence_threshold",
            "scope": "both",
            "worker_env": None,
            "notes": "Per-point confidence filter after DUSt3R alignment (0–1). Set 0 to keep all points when sparse outputs occur.",
        },
        {
            "key": "colmap_binary_path",
            "scope": "stored_on_api",
            "worker_env": None,
            "notes": "Required when reconstruction_backend=colmap: path to COLMAP executable/batch on the machine running reconstruction (for example C:\\COLMAP\\COLMAP.bat on Windows worker).",
        },
        {
            "key": "colmap_sift_gpu",
            "scope": "both",
            "worker_env": None,
            "notes": "When true: ask COLMAP to use GPU SIFT extraction when supported by your COLMAP build.",
        },
        {
            "key": "compare_mesh_preview_with_gs",
            "scope": "both",
            "worker_env": None,
            "notes": "When backend is gaussian_splatting: also write job_id_compare_mesh.glb (MapAnything mesh) before the PLY.",
        },
        {
            "key": "device",
            "scope": "both",
            "worker_env": "POLYGRAPH_OVERRIDE_DEVICE (optional)",
            "notes": "PyTorch device for SAM/MapAnything/GS. Worker defaults to cuda when a GPU is present unless overridden.",
        },
        {
            "key": "gpu_isolate_phases",
            "scope": "both",
            "worker_env": "POLYGRAPH_GPU_ISOLATE_PHASES (optional 0/false disables)",
            "notes": "When true (default on CUDA): SAM and GS scene prep run in subprocesses so VRAM drops between peaks — recommended on single 8 GB GPUs.",
        },
        {
            "key": "skip_sam_segmentation",
            "scope": "both",
            "worker_env": None,
            "notes": "When false (**default**): phase-1 isolation runs first (`isolation_backend`: sam/rembg), then MapAnything only sees black-backed masked views. When true: no automatic isolation — every upload must be **RGBA with real transparency** (cutout matte); plain RGB/JPEG or fully opaque alpha is **rejected** so background never enters silently.",
        },
        {
            "key": "isolation_backend",
            "scope": "both",
            "worker_env": None,
            "notes": "Phase-1 foreground isolator when skip_sam_segmentation=false. `sam` = promptable Segment Anything; `rembg` = local alpha-matte model (often better on cluttered textiles).",
        },
        {
            "key": "isolation_smart_select",
            "scope": "both",
            "worker_env": None,
            "notes": "When true: score multiple isolation candidates and keep the most plausible centered object mask (helps reject unrelated background/table fragments).",
        },
        {
            "key": "isolation_try_alternate_backend",
            "scope": "both",
            "worker_env": None,
            "notes": "When true: also evaluates the other backend as fallback candidate (`sam` <-> `rembg`) before accepting phase-1 mask.",
        },
        {
            "key": "isolation_min_score",
            "scope": "both",
            "worker_env": None,
            "notes": "Minimum smart-selection quality score accepted for a mask; lower values trigger stricter fallback behaviour.",
        },
        {
            "key": "recenter_isolated_subject",
            "scope": "both",
            "worker_env": None,
            "notes": "After phase-1 isolation, translate each masked view so the foreground centroid is in the image center before alignment/reconstruction.",
        },
        {
            "key": "recenter_target_subject_fill",
            "scope": "both",
            "worker_env": None,
            "notes": "Target object size before saving masked images (max foreground bbox side / image min side). Higher makes the isolated object larger in-frame.",
        },
        {
            "key": "recenter_max_scale",
            "scope": "both",
            "worker_env": None,
            "notes": "Maximum zoom applied during recentering; prevents extreme enlargement/cropping on tiny masks.",
        },
        {
            "key": "recenter_allow_per_image_scaling",
            "scope": "both",
            "worker_env": None,
            "notes": "When false (recommended): recenter only translates and preserves native object scale across views. Set true to also scale each image independently toward target fill.",
        },
        {
            "key": "isolation_filter_outlier_views",
            "scope": "both",
            "worker_env": None,
            "notes": "When true: drop clearly inconsistent masked views (abnormal object area/center drift) before reconstruction so one bad isolation does not poison the full job.",
        },
        {
            "key": "isolation_outlier_area_mad_scale",
            "scope": "both",
            "worker_env": None,
            "notes": "Robust strictness for segmentation outlier filtering by object area (median-absolute-deviation scale). Higher values keep more views.",
        },
        {
            "key": "isolation_outlier_center_distance",
            "scope": "both",
            "worker_env": None,
            "notes": "Maximum normalized distance of isolated object center from image center before view is treated as outlier.",
        },
        {
            "key": "segmentation_fail_fast",
            "scope": "both",
            "worker_env": None,
            "notes": "When true: fail the job if phase-1 cannot produce a plausible object mask, instead of silently using placeholder ellipse fallback.",
        },
        {
            "key": "rembg_model_name",
            "scope": "both",
            "worker_env": None,
            "notes": "Only when isolation_backend=rembg. Recommended starting point: `isnet-general-use`; alternatives include `u2net` / `u2netp` / `birefnet-general`.",
        },
        {
            "key": "rembg_alpha_threshold",
            "scope": "both",
            "worker_env": None,
            "notes": "Only when isolation_backend=rembg. Alpha cutoff (0–255) when converting matte to binary mask before centre-subject refinement. Raise to suppress halos.",
        },
        {
            "key": "expose_masked_views",
            "scope": "both",
            "worker_env": None,
            "notes": "After SAM/precut: isolated ``masked_*`` RGB copies appear under ``uploads/{job_id}/masked_views/`` for GET (remote worker POST ``/internal/worker/jobs/{id}/masked-views``). Mesh/GS still consume ``masked/`` paths internally. Browse thumbnails at ``GET /jobs/{job_id}/masked-preview`` (HTML) or use ``masked_view_urls`` from ``GET /jobs/{job_id}``.",
        },
        {
            "key": "mapanything_alpha_flatten_gray",
            "scope": "both",
            "worker_env": None,
            "notes": "Used only with ``skip_sam_segmentation`` and RGBA inputs: neutral grey (0–255, same on R/G/B) composited where alpha≈0 before MapAnything sees the view.",
        },
        {
            "key": "mapanything_apply_internal_mask_on_precut",
            "scope": "both",
            "worker_env": None,
            "notes": "Only when ``skip_sam_segmentation``: if false (default), MapAnything ``apply_mask``/``mask_edges`` are off — better for glossy/small toys on grey cutouts; set true when depth is fused with cluttered background pixels.",
        },
        {
            "key": "sam_checkpoint_path",
            "scope": "stored_on_api",
            "worker_env": "POLYGRAPH_OVERRIDE_SAM_CHECKPOINT",
            "notes": "Must exist on the machine that runs SAM (worker path in split deploy). Ignored when ``skip_sam_segmentation`` is true. ``sam_model_type`` must match the file: vit_b for sam_vit_b_*.pth, vit_h for sam_vit_h_*.pth, vit_l for sam_vit_l_*.pth — mismatch causes weight shape errors.",
        },
        {
            "key": "gs_repo_path",
            "scope": "stored_on_api",
            "worker_env": "POLYGRAPH_OVERRIDE_GS_REPO",
            "notes": "graphdeco-inria/gaussian-splatting clone on GPU worker.",
        },
        {
            "key": "gs_allow_cpu_fallback",
            "scope": "both",
            "worker_env": "POLYGRAPH_GS_ALLOW_CPU_FALLBACK (0/false forces GPU train path)",
            "notes": "Set false on a strict GPU worker so GS does not use CPU PLY fallback.",
        },
        {
            "key": "mesh_photo_vertex_bake",
            "scope": "both",
            "worker_env": None,
            "notes": "When true: optional photo projection onto mesh vertices after Poisson.",
        },
        {
            "key": "mesh_photo_vertex_bake_sample_source",
            "scope": "both",
            "worker_env": None,
            "notes": "`original` (default): sample full uploads for richer photo realism on the object surface. Use `masked` when you see backdrop color bleed onto vertices.",
        },
        {
            "key": "texture_surface_abstraction_enabled",
            "scope": "both",
            "worker_env": None,
            "notes": "When true: preprocess texture source images before photo-vertex bake to flatten illumination/shadow gradients while preserving surface details.",
        },
        {
            "key": "texture_surface_abstraction_strength",
            "scope": "both",
            "worker_env": None,
            "notes": "Blend amount (0..1) for abstracted texture images used in photo bake. Higher values reduce lighting artifacts more aggressively.",
        },
        {
            "key": "texture_surface_detail_preserve",
            "scope": "both",
            "worker_env": None,
            "notes": "How much high-frequency detail from original luminance is re-injected after illumination flattening (0..1).",
        },
        {
            "key": "texture_surface_illumination_blur",
            "scope": "both",
            "worker_env": None,
            "notes": "Gaussian blur kernel used to estimate illumination field before abstraction (odd integer; larger = smoother lighting removal).",
        },
        {
            "key": "surface_region_texture_enabled",
            "scope": "both",
            "worker_env": None,
            "notes": "When true: extract dominant smooth surface region from each texture source image (for example one side panel) before photo projection.",
        },
        {
            "key": "surface_region_smooth_percentile",
            "scope": "both",
            "worker_env": None,
            "notes": "Percentile threshold of gradient magnitude used to pick smooth candidate pixels for dominant surface extraction.",
        },
        {
            "key": "surface_region_min_area_ratio",
            "scope": "both",
            "worker_env": None,
            "notes": "Minimum area ratio required for extracted dominant surface region; below this, full foreground is used as fallback.",
        },
        {
            "key": "surface_region_expand_px",
            "scope": "both",
            "worker_env": None,
            "notes": "Dilation pixels applied to dominant surface region to include nearby edges/details before projection.",
        },
        {
            "key": "sam_model_type",
            "scope": "both",
            "worker_env": None,
            "notes": "vit_b / vit_l / vit_h must match the Segment Anything ``.pth`` filename you load; default vit_b matches common ``sam_vit_b_*.pth`` installs.",
        },
        {
            "key": "sam_use_fp16",
            "scope": "both",
            "worker_env": None,
            "notes": "CUDA AMP fp16 for SAM forward passes (lower VRAM on small GPUs).",
        },
        {
            "key": "sam_segmentation_mode",
            "scope": "both",
            "worker_env": None,
            "notes": (
                "Default **center_subject_table**: same as center_subject (centre FG, corners BG) plus extra "
                "**background** prompts along the **bottom edge** (table plane) so SAM drops the tabletop under "
                "the object more often. Use **center_subject** if that strip wrongly eats thin bases/feet; "
                "**center_point** is centre-only (weakest BG/table separation). Place the subject near the optical centre."
            ),
        },
        {
            "key": "sam_table_edge_negative_points",
            "scope": "both",
            "worker_env": None,
            "notes": "Only for center_subject_table: how many SAM negative clicks span the bottom inset strip (0–24). Higher = stronger table suppression; 0 matches corner-only layout.",
        },
        {
            "key": "sam_prompt_min_mask_area_ratio",
            "scope": "both",
            "worker_env": None,
            "notes": "Prompt-SAM sanity lower bound for foreground area ratio. If selected mask is smaller, it is treated as likely noise and can trigger auto recovery.",
        },
        {
            "key": "sam_prompt_max_mask_area_ratio",
            "scope": "both",
            "worker_env": None,
            "notes": "Prompt-SAM sanity upper bound for foreground area ratio. Lower this for tiny centred toys on busy fabrics when SAM keeps background instead of the object.",
        },
        {
            "key": "sam_prompt_require_center_hit",
            "scope": "both",
            "worker_env": None,
            "notes": "When true (default): prompt-mode mask selection only accepts candidates containing the image centre; otherwise it falls back to auto center-bias recovery.",
        },
        {
            "key": "sam_recover_with_auto_if_prompt_bad",
            "scope": "both",
            "worker_env": None,
            "notes": "When true (default): if prompt-SAM mask is outside min/max area bounds, rerun center-biased auto-mask selection and prefer tighter subject masks.",
        },
        {
            "key": "max_input_image_side",
            "scope": "both",
            "worker_env": None,
            "notes": "Longest edge cap when a job starts (before SAM/precut prep). Preserve alpha via PNG when resizing RGBA uploads.",
        },
        {
            "key": "gs_densify_until_iter",
            "scope": "both",
            "worker_env": None,
            "notes": "Forwarded to gaussian-splatting train.py --densify_until_iter; 0 omits flag (upstream default).",
        },
        {
            "key": "gs_train_with_original_images",
            "scope": "both",
            "worker_env": None,
            "notes": "When true (default): before GPU train.py, scene/images RGB is replaced per view with original (unmasked) photos while keeping masked filenames — reduces dark/black Gaussians from SAM black padding supervising the splat optimiser.",
        },
        {
            "key": "mesh_glb_draco_compression",
            "scope": "both",
            "worker_env": None,
            "notes": "Mesh (.glb): Open3D compressed export when true (smaller files for web).",
        },
        {
            "key": "cdn_base_url",
            "scope": "api",
            "worker_env": None,
            "notes": "Public URL prefix for output files served by the API.",
        },
        {
            "key": "max_images",
            "scope": "both",
            "worker_env": None,
            "notes": "Job upload limit.",
        },
    ]
    return {
        "title": "Where each setting applies",
        "summary": (
            "PUT /settings stores one JSON document. The API uses it for persistence and passes a copy to "
            "remote GPU workers with each job. Paths like sam_checkpoint_path often point at the API disk; "
            "use POLYGRAPH_OVERRIDE_* in .env.worker so the worker resolves real files on the GPU machine."
        ),
        "worker_only_env": [
            {"name": "POLYGRAPH_API_BASE", "purpose": "HTTPS root of the API (same host as uploads)."},
            {"name": "POLYGRAPH_WORKER_TOKEN", "purpose": "Must match APP_WORKER_TOKEN on the API."},
            {"name": "POLYGRAPH_REQUIRE_CUDA", "purpose": "If 1, worker exits when torch.cuda.is_available() is false."},
            {"name": "POLYGRAPH_OVERRIDE_MAPANYTHING_MODEL", "purpose": "Optional override for mapanything_pretrained_id (HF id)."},
            {"name": "POLYGRAPH_OVERRIDE_DUST3R_CHECKPOINT", "purpose": "Optional worker-local override for dust3r_checkpoint_path."},
            {"name": "POLYGRAPH_OVERRIDE_AI_PRIOR_COMMAND", "purpose": "Optional worker-local override for ai_prior_command."},
            {"name": "POLYGRAPH_OVERRIDE_TRIPOSR_REPO", "purpose": "Optional worker-local override for ai_prior_triposr_repo_path."},
            {"name": "POLYGRAPH_OVERRIDE_TRIPOSR_PYTHON", "purpose": "Optional worker-local override for ai_prior_triposr_python_executable."},
            {"name": "POLYGRAPH_OVERRIDE_INSTANTMESH_REPO", "purpose": "Optional worker-local override for ai_prior_instantmesh_repo_path."},
            {"name": "POLYGRAPH_OVERRIDE_INSTANTMESH_PYTHON", "purpose": "Optional worker-local override for ai_prior_instantmesh_python_executable."},
            {"name": "POLYGRAPH_OVERRIDE_FFMPEG_BINARY", "purpose": "Optional worker-local override for video_ffmpeg_binary path."},
            {"name": "POLYGRAPH_AI_PRIOR_UPSTREAM_CMD", "purpose": "Real provider command used by scripts/ai_prior_command_adapter.py."},
            {"name": "POLYGRAPH_AI_PRIOR_UPSTREAM_ARGS_TEMPLATE", "purpose": "Optional args template for upstream provider command."},
            {"name": "TRIPO_API_KEY", "purpose": "Tripo OpenAPI key used by scripts/tripo_api_upstream.py when testing cloud generation."},
            {"name": "TRIPO_MODEL_VERSION", "purpose": "Optional Tripo image-to-model version (example v3.1-20260211)."},
        ],
        "fields": fields,
        "limitations": (
            "Open3D mesh steps are CPU-bound. CUDA is used for SAM, MapAnything inference, and GS train.py on GPU workers."
        ),
    }
