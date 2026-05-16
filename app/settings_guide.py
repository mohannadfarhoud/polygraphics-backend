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
            "notes": "Mesh: mapanything (.glb); gaussian_splatting (.ply) — usually closer to photos for glossy objects when trained on GPU.",
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
            "notes": "When false (default for full-camera photos): SAM segments the centred subject so the tabletop/room can stay black-backed. Requires sam_checkpoint_path on the worker unless allow_placeholder_pipeline. When true: you must upload pre-cut PNGs whose **opaque pixels are only the object** — the table cannot be painted out later.",
        },
        {
            "key": "expose_masked_views",
            "scope": "both",
            "worker_env": None,
            "notes": "After SAM/precut: isolated ``masked_*`` RGB copies appear under ``uploads/{job_id}/masked_views/`` for GET (remote worker POST ``/internal/worker/jobs/{id}/masked-views``). Mesh/GS still consume ``masked/`` paths internally.",
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
            "notes": "`masked`: sample SAM black-background crops (recommended — avoids backdrop colours on vertices). `original`: sample full uploads (legacy; can smear clutter onto the mesh).",
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
            "notes": "For a **toy centred in frame on a table**: use skip_sam_segmentation=false with center_subject — centre = foreground prompt, corners = background prompts. center_point only centre tap (weaker separation from tabletop). Place the car near the optical centre so the centroid matches the SAM prompt.",
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
        ],
        "fields": fields,
        "limitations": (
            "Open3D mesh steps are CPU-bound. CUDA is used for SAM, MapAnything inference, and GS train.py on GPU workers."
        ),
    }
