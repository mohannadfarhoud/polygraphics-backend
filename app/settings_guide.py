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
            "notes": "Primary pipeline: mesh (dust3r/colmap/auto) or gaussian_splatting (.ply).",
        },
        {
            "key": "compare_mesh_dust3r_colmap_with_gs",
            "scope": "both",
            "worker_env": None,
            "notes": "When backend is gaussian_splatting, also write job_id_compare_dust3r.glb and job_id_compare_colmap.glb before the PLY.",
        },
        {
            "key": "device",
            "scope": "both",
            "worker_env": "POLYGRAPH_OVERRIDE_DEVICE (optional)",
            "notes": "PyTorch device for SAM/DUSt3R/GS. Worker defaults to cuda when a GPU is present unless overridden.",
        },
        {
            "key": "sam_checkpoint_path",
            "scope": "stored_on_api",
            "worker_env": "POLYGRAPH_OVERRIDE_SAM_CHECKPOINT",
            "notes": "Must exist on the machine that runs SAM (worker path in split deploy).",
        },
        {
            "key": "dust3r_checkpoint_path",
            "scope": "stored_on_api",
            "worker_env": "POLYGRAPH_OVERRIDE_DUST3R_CHECKPOINT",
            "notes": "Must exist on the worker for remote jobs.",
        },
        {
            "key": "dust3r_repo_path",
            "scope": "stored_on_api",
            "worker_env": "POLYGRAPH_OVERRIDE_DUST3R_REPO",
            "notes": "DUSt3R clone path on the worker.",
        },
        {
            "key": "colmap_binary_path",
            "scope": "stored_on_api",
            "worker_env": "POLYGRAPH_OVERRIDE_COLMAP_PATH",
            "notes": "COLMAP executable on the worker.",
        },
        {
            "key": "colmap_sift_gpu",
            "scope": "both",
            "worker_env": None,
            "notes": "COLMAP feature_extractor SiftExtraction.use_gpu when true (requires CUDA COLMAP build).",
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
            "key": "mesh_texture_mapping",
            "scope": "both",
            "worker_env": "POLYGRAPH_MESH_TEXTURE_MAPPING",
            "notes": "UV atlas bake (CPU-heavy). Ignored for gaussian_splatting main output.",
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
            {"name": "POLYGRAPH_MESH_TEXTURE_MAPPING", "purpose": "1/0 overrides mesh_texture_mapping; unset uses PUT /settings value."},
            {"name": "POLYGRAPH_WS_PING_TIMEOUT", "purpose": "WebSocket keepalive for long jobs."},
        ],
        "fields": fields,
        "limitations": (
            "Some steps are inherently CPU-only in this codebase (Open3D Poisson/decimate, xatlas texture bake, "
            "parts of COLMAP). CUDA is used for SAM, DUSt3R, COLMAP SIFT when enabled, and GS train.py on GPU."
        ),
    }
