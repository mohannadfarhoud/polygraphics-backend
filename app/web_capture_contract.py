from __future__ import annotations

from typing import Any

from .runtime_settings import RuntimeSettings, effective_reconstruction_backend, minimum_input_images


def _recommended_capture_counts(settings: RuntimeSettings) -> dict[str, int]:
    max_images = max(2, int(getattr(settings, "max_images", 100)))
    hard_min = max(1, int(minimum_input_images(settings)))
    # Even with single-image backends, web capture quality is better when users provide
    # enough views for filtering and best-view selection before prior generation.
    warn_below = min(max_images, max(12, hard_min))
    target_min = min(max_images, max(24, warn_below))
    target_max = min(max_images, max(target_min, 36))
    return {
        "hard_minimum": hard_min,
        "warn_below": warn_below,
        "target_min": target_min,
        "target_max": target_max,
    }


def build_web_capture_contract(settings: RuntimeSettings) -> dict[str, Any]:
    backend = str(effective_reconstruction_backend(settings)).strip().lower()
    provider = str(getattr(settings, "ai_prior_provider", "")).strip().lower()
    force_prior_only = bool(getattr(settings, "ai_prior_force_prior_only", False))
    counts = _recommended_capture_counts(settings)

    return {
        "schema_version": 1,
        "mode": "web_capture",
        "summary": (
            "Machine-readable capture rules for web UI. Values are derived from current runtime settings "
            "so frontend guidance matches backend filtering behavior."
        ),
        "pipeline": {
            "reconstruction_backend": backend,
            "ai_prior_provider": provider or None,
            "ai_prior_force_prior_only": force_prior_only,
            "minimum_input_images": counts["hard_minimum"],
        },
        "capture_targets": {
            "image_count": counts,
            "object_fill_ratio_hint": {
                "recommended_min": 0.60,
                "recommended_max": 0.75,
                "note": "Keep object size stable across views to improve depth consistency and best-view selection.",
            },
            "orbit_coverage_hint_deg": {
                "recommended_min": 220.0,
                "recommended_target": 300.0,
            },
        },
        "live_frame_checks": {
            "blur": {
                "metric": "laplacian_variance",
                "formula": "var(cv2.Laplacian(gray, cv2.CV_64F))",
                "min": float(getattr(settings, "capture_blur_min", 35.0)),
            },
            "brightness": {
                "metric": "gray_mean_0_255",
                "formula": "mean(gray)",
                "min": float(getattr(settings, "capture_brightness_min", 20.0)),
                "max": float(getattr(settings, "capture_brightness_max", 235.0)),
            },
            "frame_delta": {
                "metric": "mean_abs_diff_64x64",
                "formula": "mean(abs(curr_tiny - prev_tiny)), tiny=resize(gray,64x64)/255",
                "min": float(getattr(settings, "capture_min_frame_delta", 0.010)),
            },
            "duplicate_similarity": {
                "metric": "cosine_similarity_64x64",
                "formula": "dot(curr_tiny, kept_tiny) / (norm(curr_tiny)*norm(kept_tiny))",
                "max": float(getattr(settings, "capture_duplicate_similarity", 0.995)),
            },
            "diversity_distance": {
                "metric": "mean_abs_diff_64x64",
                "formula": "min(mean(abs(curr_tiny - selected_tiny_i))) across selected frames",
                "min": float(getattr(settings, "capture_diversity_min_distance", 0.045)),
            },
        },
        "selection_policy": {
            "capture_reject_policy": str(getattr(settings, "capture_reject_policy", "soft")).strip().lower(),
            "capture_min_kept_images": int(getattr(settings, "capture_min_kept_images", 8)),
            "capture_max_selected_images": int(getattr(settings, "capture_max_selected_images", 20)),
            "depth_normalization_enabled": bool(getattr(settings, "depth_normalization_enabled", True)),
            "depth_reference_mode": str(getattr(settings, "depth_reference_mode", "median_best")).strip().lower(),
            "depth_consistency_apply_filter": bool(getattr(settings, "depth_consistency_apply_filter", True)),
            "depth_consistency_max_relative": float(getattr(settings, "depth_consistency_max_relative", 0.35)),
            "depth_consistency_min_kept_images": int(getattr(settings, "depth_consistency_min_kept_images", 3)),
        },
        "upload_constraints": {
            "max_images": int(getattr(settings, "max_images", 100)),
            "max_input_image_side": int(getattr(settings, "max_input_image_side", 1920)),
            "supported_upload_endpoints": ["/jobs", "/jobs/reconstruct"],
        },
        "capture_metadata_contract": {
            "upload_field_name": "capture_metadata",
            "post_upload_endpoint": "/jobs/{job_id}/capture-metadata",
            "frame_fields": [
                "file",
                "frame_index",
                "timestamp_ms",
                "yaw_deg",
                "pitch_deg",
                "roll_deg",
                "distance_m",
                "depth_m",
                "blur_score",
                "exposure_score",
                "quality_score",
                "accepted",
            ],
            "summary_fields": [
                "total_frames",
                "accepted_frames",
                "rejected_frames",
                "avg_quality_score",
                "orbit_coverage_deg",
            ],
        },
        "report_files": [
            "uploads/{job_id}/capture_quality_report.json",
            "uploads/{job_id}/depth_normalization_report.json",
            "uploads/{job_id}/reconstruction_report.json",
        ],
    }

