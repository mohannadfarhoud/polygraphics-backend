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
            "notes": "Mesh: mapanything (.glb), dust3r (.glb), or colmap (.glb); gaussian_splatting (.ply) for splats.",
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
        ],
        "fields": fields,
        "limitations": (
            "Open3D mesh steps are CPU-bound. CUDA is used for SAM, MapAnything inference, and GS train.py on GPU workers."
        ),
    }
