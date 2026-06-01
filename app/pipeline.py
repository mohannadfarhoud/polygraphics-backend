from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import open3d as o3d

from .color_baking import CameraView
from .config import PipelineConfig
from .interfaces import JobRepository, JobStatus, NoopJobRepository, NoopWebSocketNotifier, WebSocketNotifier
from .meshing import (
    autobalance_vertex_colors,
    center_and_scale_mesh,
    decimate,
    export_glb,
    keep_largest_mesh_component,
    poisson_mesh,
    transfer_vertex_colors_from_point_cloud,
)
from .cuda_memory import effective_gpu_isolate_phases, purge_torch_cuda
from .pipeline_ready import assert_pipeline_ready
from .point_cloud import build_point_cloud, remove_statistical_outliers
from .reconstruction import Dust3RReconstructor
from .runtime_settings import (
    RuntimeSettings,
    effective_reconstruction_backend,
    gaussian_splatting_skipped_via_env,
    minimum_input_images,
    resolve_auto_backend,
)
from .segmentation import SamSegmenter

_log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]


class JobCancelled(Exception):
    """Raised when cancel_event is set (stop/pause)."""


class ReconstructionPipeline:
    def __init__(
        self,
        config: PipelineConfig,
        runtime_settings: RuntimeSettings,
        *,
        segmenter: SamSegmenter | None = None,
        reconstructor: Dust3RReconstructor | None = None,
        job_repo: JobRepository | None = None,
        notifier: WebSocketNotifier | None = None,
    ) -> None:
        self.config = config
        self.runtime_settings = runtime_settings
        self.segmenter = segmenter or SamSegmenter(runtime_settings)
        self.reconstructor = reconstructor or Dust3RReconstructor(runtime_settings)
        self.job_repo = job_repo or NoopJobRepository()
        self.notifier = notifier or NoopWebSocketNotifier()

    def process_3d_job(
        self,
        job_id: str,
        image_paths: list[Path],
        *,
        cancel_event: threading.Event | None = None,
    ) -> str:
        self._publish(job_id, JobStatus.PROCESSING, stage="starting", progress=5)

        try:
            assert_pipeline_ready(self.runtime_settings)
            self._raise_if_cancelled(cancel_event)
            if (
                gaussian_splatting_skipped_via_env()
                and self.runtime_settings.reconstruction_backend == "gaussian_splatting"
            ):
                _log.warning(
                    "POLYGRAPH_SKIP_GAUSSIAN_SPLATTING is set — running MapAnything mesh path instead of Gaussian Splatting"
                )

            from .image_preprocess import downscale_job_images_if_needed
            from .mapanything_input_prep import prepare_precut_opaque_views_for_mapanything

            _VIDEO_EXTS = frozenset({".mp4", ".mov", ".webm", ".avi", ".mkv"})
            video_input_mode = bool(getattr(self.runtime_settings, "video_input_enabled", False))

            # ------------------------------------------------------------------
            # Video input mode: extract frames FIRST so the full quality
            # pipeline operates on real image frames, not the video file.
            # If video mode is off, strip any stray video files from the list.
            # ------------------------------------------------------------------
            if video_input_mode:
                from .video_frame_extractor import find_video_in_upload_dir, run_video_extraction

                self._publish(
                    job_id, JobStatus.PROCESSING,
                    stage="phase_video_extraction", progress=6,
                )
                # Accept either an explicit video upload or, if the "image" paths
                # are just a video file forwarded by the worker, detect it there.
                video_path = find_video_in_upload_dir(
                    self.config.root_dir / "uploads", job_id
                )
                if video_path is None:
                    # Fallback: first path passed in might be the video itself
                    for p in image_paths:
                        if p.suffix.lower() in _VIDEO_EXTS:
                            video_path = p
                            break
                if video_path is None:
                    raise RuntimeError(
                        "video_input_enabled=true but no video file found under "
                        f"uploads/{job_id}/. Upload a .mp4/.mov/.webm file."
                    )
                extraction = run_video_extraction(
                    job_id=job_id,
                    video_path=video_path,
                    upload_dir=self.config.root_dir / "uploads",
                    fps=float(getattr(self.runtime_settings, "video_input_extraction_fps", 2.0)),
                    min_sharpness=float(getattr(self.runtime_settings, "video_frame_min_sharpness", 0.04)),
                    min_exposure=float(getattr(self.runtime_settings, "video_frame_min_exposure", 0.10)),
                    max_frames=int(getattr(self.runtime_settings, "video_input_max_frames", 60)),
                    max_duration_seconds=int(getattr(self.runtime_settings, "video_input_max_duration_seconds", 30)),
                    ffmpeg_bin=getattr(self.runtime_settings, "video_ffmpeg_binary", None) or None,
                )
                self._publish(
                    job_id, JobStatus.PROCESSING,
                    stage=f"phase_video_frame_selection ({extraction.total_kept}/{extraction.total_extracted} frames kept)",
                    progress=9,
                )
                image_paths = extraction.kept_frames if extraction.kept_frames else extraction.all_frames
                _log.info(
                    "job=%s: video extracted=%d, quality-kept=%d → these will drive backend auto-selection",
                    job_id, extraction.total_extracted, len(image_paths),
                )
            else:
                # Safety: filter out any video files passed as images
                image_paths = [p for p in image_paths if p.suffix.lower() not in _VIDEO_EXTS]

            image_paths = downscale_job_images_if_needed(
                job_id,
                image_paths,
                self.config.root_dir / "data" / "job_inputs",
                int(self.runtime_settings.max_input_image_side),
            )
            self._publish(job_id, JobStatus.PROCESSING, stage="phase_0_input_curation", progress=8)
            quality_score = 1.0
            preferred_prior_input: Path | None = None
            try:
                from .capture_quality import run_capture_quality_gate

                quality = run_capture_quality_gate(
                    job_id=job_id,
                    image_paths=image_paths,
                    settings=self.runtime_settings,
                    upload_dir=self.config.root_dir / "uploads",
                )
                image_paths = quality.kept_paths
                quality_score = float(quality.score)
            except Exception:
                # Let hard policy errors propagate; only ignore unexpected telemetry failures.
                if str(self.runtime_settings.capture_reject_policy).strip().lower() == "hard":
                    raise
            if len(image_paths) < 1:
                raise RuntimeError(
                    "No usable images left after quality filtering. "
                    "Please retake with better lighting and a steadier hand."
                )
            raw_backend = str(effective_reconstruction_backend(self.runtime_settings)).strip().lower()
            ai_provider = str(getattr(self.runtime_settings, "ai_prior_provider", "")).strip().lower()

            # ------------------------------------------------------------------
            # Auto backend selection: resolve "auto" → concrete backend based
            # on how many images survived the quality gate.
            # ------------------------------------------------------------------
            if raw_backend == "auto":
                backend, auto_ai_provider = resolve_auto_backend(
                    self.runtime_settings, len(image_paths)
                )
                if auto_ai_provider is not None:
                    ai_provider = auto_ai_provider
                _log.info(
                    "job=%s: auto backend selected %r (provider=%r) for %d image(s)",
                    job_id, backend, ai_provider, len(image_paths),
                )
            else:
                backend = raw_backend

            # Single-image AI models (TripoSR / InstantMesh): SAM runs normally to produce clean
            # masked images; depth normalization is skipped (not useful for single-image models).
            triposr_local_mode = backend == "ai_prior" and ai_provider == "triposr_local"
            instantmesh_local_mode = backend == "ai_prior" and ai_provider == "instantmesh_local"
            single_image_ai_mode = triposr_local_mode or instantmesh_local_mode

            if self.runtime_settings.skip_sam_segmentation:
                # Pre-cut uploads only (RGBA + alpha matte); see prepare_precut_opaque_views_for_mapanything.
                self._publish(
                    job_id,
                    JobStatus.PROCESSING,
                    stage="precut_opaque_views",
                    progress=28,
                )
                masked_paths = prepare_precut_opaque_views_for_mapanything(
                    job_id,
                    image_paths,
                    masked_dir=self.config.masked_dir,
                    flatten_gray_0_255=int(self.runtime_settings.mapanything_alpha_flatten_gray),
                )
                originals_for_mesh = list(masked_paths)
            else:
                # Standard path: SAM masks full-frame photos before any MapAnything / mesh work.
                masked_paths = self._run_segmentation(job_id, image_paths, cancel_event=cancel_event)
                originals_for_mesh = image_paths
            depth_fail_hard = bool(getattr(self.runtime_settings, "depth_consistency_fail_on_low_score", False))
            depth_score = 1.0
            if not single_image_ai_mode:
                try:
                    from .depth_normalization import run_depth_normalization_gate

                    self._publish(job_id, JobStatus.PROCESSING, stage="phase_depth_normalization", progress=40)
                    depth_norm = run_depth_normalization_gate(
                        job_id=job_id,
                        masked_paths=masked_paths,
                        original_paths=originals_for_mesh,
                        settings=self.runtime_settings,
                        upload_dir=self.config.root_dir / "uploads",
                    )
                    masked_paths = depth_norm.masked_paths
                    originals_for_mesh = depth_norm.original_paths
                    preferred_prior_input = depth_norm.selected_masked_path
                    depth_score = float(depth_norm.consistency_score)
                    # Blend capture + depth consistency so route confidence reflects both.
                    quality_score = float(max(0.0, min(1.0, (0.75 * quality_score) + (0.25 * depth_score))))
                    min_depth_score = float(getattr(self.runtime_settings, "depth_consistency_min_score", 0.40))
                    if depth_norm.applied and depth_fail_hard and depth_score < min_depth_score:
                        raise RuntimeError(
                            f"Depth consistency too low ({depth_score:.3f} < {min_depth_score:.3f}). "
                            "Retake with more stable camera distance around the object."
                        )
                except Exception as exc:
                    if depth_fail_hard:
                        raise
                    _log.warning("depth normalization stage failed job=%s: %s", job_id, exc)
            try:
                self.segmenter.release_gpu_memory()
            except Exception:
                pass
            purge_torch_cuda()
            self._raise_if_cancelled(cancel_event)
            if backend == "gaussian_splatting":
                if self.runtime_settings.compare_mesh_preview_with_gs:
                    self._publish(
                        job_id,
                        JobStatus.PROCESSING,
                        stage="compare_mesh_preview",
                        progress=42,
                    )
                    self._run_mesh_pipeline(
                        job_id,
                        masked_paths,
                        originals_for_mesh,
                        cancel_event=cancel_event,
                        mesh_backend="mapanything",
                        output_basename=f"{job_id}_compare_mesh",
                        publish_completed=False,
                    )
                    self._raise_if_cancelled(cancel_event)
                model_url = self._run_gaussian_splatting(
                    job_id, masked_paths, originals_for_mesh, cancel_event=cancel_event
                )
                return model_url

            if backend == "ai_prior" and instantmesh_local_mode:
                return self._run_instantmesh_pipeline(
                    job_id=job_id,
                    masked_paths=masked_paths,
                    original_paths=originals_for_mesh,
                    cancel_event=cancel_event,
                )
            if backend == "ai_prior":
                # Pass resolved ai_provider so auto-selected TripoSR is actually used.
                _provider_override = ai_provider if raw_backend == "auto" else None
                return self._run_ai_prior_pipeline(
                    job_id=job_id,
                    masked_paths=masked_paths,
                    original_paths=originals_for_mesh,
                    preferred_prior_input=preferred_prior_input,
                    quality_score=quality_score,
                    cancel_event=cancel_event,
                    provider_override=_provider_override,
                )
            if backend == "hybrid_prior_refine":
                return self._run_hybrid_prior_pipeline(
                    job_id=job_id,
                    masked_paths=masked_paths,
                    original_paths=originals_for_mesh,
                    preferred_prior_input=preferred_prior_input,
                    quality_score=quality_score,
                    cancel_event=cancel_event,
                )

            model_url = self._run_mesh_pipeline(
                job_id, masked_paths, originals_for_mesh, cancel_event=cancel_event
            )
            return model_url
        except JobCancelled:
            raise
        except Exception as exc:
            self._publish(job_id, JobStatus.FAILED, error=str(exc))
            raise

    def _write_reconstruction_report(self, job_id: str, payload: dict) -> None:
        out = self.config.root_dir / "uploads" / job_id / "reconstruction_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _route_from_confidence(self, confidence: float) -> tuple[str, str]:
        if bool(getattr(self.runtime_settings, "ai_prior_force_prior_only", False)):
            return "forced", "prior_only"
        high = float(getattr(self.runtime_settings, "reconstruction_confidence_high_threshold", 0.72))
        low = float(getattr(self.runtime_settings, "reconstruction_confidence_min_threshold", 0.45))
        if confidence >= high:
            return "high", "hybrid_refine"
        if confidence >= low:
            return "medium", "prior_only"
        policy = str(getattr(self.runtime_settings, "reconstruction_low_confidence_policy", "prior_only"))
        policy = policy.strip().lower()
        if policy == "fail":
            return "low", "fail"
        if policy == "coarse_prior":
            return "low", "coarse_prior"
        return "low", "prior_only"

    @staticmethod
    def _refine_prior_mesh_with_points(
        mesh: o3d.geometry.TriangleMesh,
        points_xyz: np.ndarray,
        *,
        strength: float,
    ) -> o3d.geometry.TriangleMesh:
        if points_xyz is None or len(points_xyz) < 100:
            return mesh
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        if verts.size == 0:
            return mesh
        pts = np.asarray(points_xyz, dtype=np.float64)
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(pts)
            try:
                _, idx = tree.query(verts, k=1, workers=-1)
            except TypeError:
                _, idx = tree.query(verts, k=1)
            idx = np.asarray(idx, dtype=np.intp).reshape(-1)
            nearest = pts[idx]
        except Exception:
            return mesh
        alpha = float(max(0.0, min(1.0, strength)))
        blended = ((1.0 - alpha) * verts) + (alpha * nearest)
        out = o3d.geometry.TriangleMesh(mesh)
        out.vertices = o3d.utility.Vector3dVector(blended)
        out.compute_vertex_normals()
        return out

    def _build_photo_views(
        self,
        reconstruction: object,
        *,
        original_paths: list[Path],
        masked_paths: list[Path],
        source_remap: dict[str, Path] | None = None,
    ) -> list[CameraView]:
        cams = list(getattr(reconstruction, "cameras", []) or [])
        if not cams:
            return []
        masked_to_original: dict[str, Path] = {}
        original_to_masked: dict[str, Path] = {}
        for masked, original in zip(masked_paths, original_paths):
            masked_to_original[str(masked)] = original
            masked_to_original[masked.name] = original
            original_to_masked[str(original)] = masked
            original_to_masked[original.name] = masked
        photo_views: list[CameraView] = []
        for cam in cams:
            if self.runtime_settings.mesh_photo_vertex_bake_sample_source == "original":
                sample_path = (
                    masked_to_original.get(str(cam.image_path))
                    or masked_to_original.get(Path(cam.image_path).name)
                    or cam.image_path
                )
            else:
                sample_path = (
                    original_to_masked.get(str(cam.image_path))
                    or original_to_masked.get(Path(cam.image_path).name)
                    or Path(cam.image_path)
                )
            remap = source_remap or {}
            try:
                sample_key_abs = str(Path(sample_path).resolve())
            except Exception:
                sample_key_abs = str(sample_path)
            sample_path = (
                remap.get(sample_key_abs)
                or remap.get(str(sample_path))
                or remap.get(Path(sample_path).name)
                or sample_path
            )
            photo_views.append(
                CameraView(
                    image_path=sample_path,
                    image_size=cam.image_size,
                    K=cam.K,
                    w2c=cam.w2c,
                )
            )
        return photo_views

    def _prepare_texture_source_remap(
        self,
        *,
        job_id: str,
        original_paths: list[Path],
        masked_paths: list[Path],
    ) -> tuple[dict[str, Path], dict[str, dict[str, object]]]:
        sample_source = str(
            getattr(self.runtime_settings, "mesh_photo_vertex_bake_sample_source", "original")
        ).strip().lower()
        base_paths = list(original_paths if sample_source == "original" else masked_paths)
        remap: dict[str, Path] = {}
        region_meta: dict[str, dict[str, object]] = {}

        # Stage 1: illumination / albedo abstraction.
        if bool(getattr(self.runtime_settings, "texture_surface_abstraction_enabled", False)):
            try:
                from .texture_abstraction import build_abstracted_texture_images

                remap = build_abstracted_texture_images(
                    job_id=job_id,
                    image_paths=base_paths,
                    cache_root=self.config.root_dir / "data" / "texture_abstraction",
                    strength=float(getattr(self.runtime_settings, "texture_surface_abstraction_strength", 0.42)),
                    detail_preserve=float(getattr(self.runtime_settings, "texture_surface_detail_preserve", 0.70)),
                    illumination_blur=int(getattr(self.runtime_settings, "texture_surface_illumination_blur", 41)),
                )
            except Exception as exc:
                _log.warning("texture surface abstraction failed job=%s: %s", job_id, exc)
                remap = {}

        # Stage 2: dominant surface extraction per view (e.g. car right side panel).
        if bool(getattr(self.runtime_settings, "surface_region_texture_enabled", False)):
            try:
                from .surface_region_texture import build_dominant_surface_texture_images

                src_paths: list[Path] = []
                for p in base_paths:
                    key = str(p.resolve())
                    src_paths.append(remap.get(key) or remap.get(str(p)) or remap.get(p.name) or p)
                self._publish(
                    job_id,
                    JobStatus.PROCESSING,
                    stage="phase_surface_region_extract",
                    progress=80,
                )
                region_result = build_dominant_surface_texture_images(
                    job_id=job_id,
                    image_paths=src_paths,
                    cache_root=self.config.root_dir / "data" / "surface_region_texture",
                    smooth_percentile=float(getattr(self.runtime_settings, "surface_region_smooth_percentile", 85.0)),
                    min_area_ratio=float(getattr(self.runtime_settings, "surface_region_min_area_ratio", 0.08)),
                    expand_px=int(getattr(self.runtime_settings, "surface_region_expand_px", 8)),
                )
                region_map = region_result.remap
                region_meta = {
                    str(entry.source_path.resolve()): {
                        "confidence": float(entry.confidence),
                        "label_hint": entry.label_hint,
                        "area_ratio": float(entry.area_ratio),
                        "failed": bool(entry.failed),
                        "failure_reason": entry.failure_reason,
                        "view_name": entry.view_name,
                    }
                    for entry in region_result.entries
                }
                if remap:
                    # Chain maps: original -> abstracted -> dominant-region.
                    chained: dict[str, Path] = {}
                    for k, v in remap.items():
                        key = str(v.resolve())
                        chained[k] = region_map.get(key) or region_map.get(str(v)) or region_map.get(v.name) or v
                    remap = chained
                else:
                    remap = region_map
            except Exception as exc:
                _log.warning("surface region texture extraction failed job=%s: %s", job_id, exc)

        return remap, region_meta

    def _run_ai_prior_pipeline(
        self,
        *,
        job_id: str,
        masked_paths: list[Path],
        original_paths: list[Path],
        preferred_prior_input: Path | None = None,
        quality_score: float,
        cancel_event: threading.Event | None = None,
        provider_override: str | None = None,
    ) -> str:
        from .ai_prior_runner import run_ai_prior_mesh

        self._publish(job_id, JobStatus.PROCESSING, stage="phase_confidence_routing", progress=41)
        band, route = self._route_from_confidence(float(quality_score))
        if route == "fail":
            raise RuntimeError(
                f"Low reconstruction confidence ({quality_score:.3f}); "
                "capture set is too weak for reliable mesh generation."
            )

        self._publish(job_id, JobStatus.PROCESSING, stage="phase_prior_generation", progress=52)
        prior = run_ai_prior_mesh(
            job_id=job_id,
            masked_images=masked_paths,
            original_images=original_paths,
            settings=self.runtime_settings,
            work_dir=self.config.root_dir / "data" / "ai_prior_workspace" / job_id,
            preferred_input_image=preferred_prior_input,
            provider_override=provider_override,
        )
        # TripoSR-local passthrough mode: return provider mesh directly for viewing
        # without any mesh cleanup/refinement/autobalance stages after generation.
        if prior.provider == "triposr_local":
            self._publish(job_id, JobStatus.PROCESSING, stage="exporting", progress=94)
            glb_path = self.config.output_dir / f"{job_id}.glb"
            src_raw = prior.details.get("output_mesh")
            src = Path(str(src_raw)).resolve() if isinstance(src_raw, str) and src_raw else None
            try:
                if src and src.is_file():
                    if src != glb_path.resolve():
                        glb_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src, glb_path)
                else:
                    export_glb(
                        prior.mesh,
                        glb_path,
                        compressed=bool(self.runtime_settings.mesh_glb_draco_compression),
                    )
            except Exception:
                export_glb(
                    prior.mesh,
                    glb_path,
                    compressed=bool(self.runtime_settings.mesh_glb_draco_compression),
                )
            model_url = f"{self.config.cdn_base_url.rstrip('/')}/{job_id}.glb"
            self._write_reconstruction_report(
                job_id,
                {
                    "job_id": job_id,
                    "reconstruction_confidence": round(float(quality_score), 4),
                    "route_taken": "prior_only",
                    "quality_reason": "triposr_local_passthrough",
                    "ai_prior_provider": prior.provider,
                    "ai_prior_confidence": round(float(prior.confidence), 4),
                    "details": prior.details,
                },
            )
            self._publish(
                job_id,
                JobStatus.COMPLETED,
                stage="completed",
                progress=100,
                model_url=model_url,
                model_format="glb",
            )
            return model_url
        mesh = keep_largest_mesh_component(prior.mesh)
        if route == "coarse_prior":
            mesh = decimate(mesh, max(10_000, int(self.config.decimation_target_triangles * 0.25)))
        mesh = center_and_scale_mesh(mesh)
        mesh = autobalance_vertex_colors(mesh)

        self._publish(job_id, JobStatus.PROCESSING, stage="exporting", progress=94)
        glb_path = self.config.output_dir / f"{job_id}.glb"
        export_glb(
            mesh,
            glb_path,
            compressed=bool(self.runtime_settings.mesh_glb_draco_compression),
        )
        model_url = f"{self.config.cdn_base_url.rstrip('/')}/{job_id}.glb"
        self._write_reconstruction_report(
            job_id,
            {
                "job_id": job_id,
                "reconstruction_confidence": round(float(quality_score), 4),
                "route_taken": "prior_only" if route != "coarse_prior" else "coarse_prior",
                "quality_reason": f"{band}_confidence_input",
                "ai_prior_provider": prior.provider,
                "ai_prior_confidence": round(float(prior.confidence), 4),
                "details": prior.details,
            },
        )
        self._publish(
            job_id,
            JobStatus.COMPLETED,
            stage="completed",
            progress=100,
            model_url=model_url,
            model_format="glb",
        )
        return model_url

    def _run_instantmesh_pipeline(
        self,
        *,
        job_id: str,
        masked_paths: list[Path],
        original_paths: list[Path],
        cancel_event: threading.Event | None = None,
    ) -> str:
        """InstantMesh single-image reconstruction pipeline.

        1. Select best (masked, original) pair using multi-factor quality score.
        2. Run InstantMesh to produce a mesh.
        3. Passthrough export — no extra post-processing.
        """
        from .instantmesh_runner import run_instantmesh
        from .ai_prior_runner import _pick_best_triposr_pair  # shared selection logic

        self._publish(job_id, JobStatus.PROCESSING, stage="phase_prior_generation", progress=52)
        best_masked, best_original = _pick_best_triposr_pair(masked_paths, original_paths)

        result = run_instantmesh(
            job_id=job_id,
            selected_frame=best_original,
            masked_frame=best_masked,
            settings=self.runtime_settings,
            work_dir=self.config.root_dir / "data" / "ai_prior_workspace" / job_id,
        )
        self._raise_if_cancelled(cancel_event)

        self._publish(job_id, JobStatus.PROCESSING, stage="exporting", progress=94)
        glb_path = self.config.output_dir / f"{job_id}.glb"
        src = result.output_mesh.resolve()
        try:
            if src != glb_path.resolve():
                glb_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, glb_path)
        except Exception:
            export_glb(
                result.mesh,
                glb_path,
                compressed=bool(self.runtime_settings.mesh_glb_draco_compression),
            )

        model_url = f"{self.config.cdn_base_url.rstrip('/')}/{job_id}.glb"

        # Expose selected_frame URL via reconstruction report
        frame_rel = str(result.selected_frame.relative_to(self.config.root_dir))
        self._write_reconstruction_report(
            job_id,
            {
                "job_id": job_id,
                "reconstruction_backend": "ai_prior",
                "ai_prior_provider": "instantmesh_local",
                "route_taken": "instantmesh_passthrough",
                "selected_frame": str(result.selected_frame),
                "selected_frame_rel": frame_rel,
                "debug_dir": str(result.debug_dir) if result.debug_dir else None,
                "details": result.details,
            },
        )
        self._publish(
            job_id,
            JobStatus.COMPLETED,
            stage="completed",
            progress=100,
            model_url=model_url,
            model_format="glb",
        )
        return model_url

    def _run_hybrid_prior_pipeline(
        self,
        *,
        job_id: str,
        masked_paths: list[Path],
        original_paths: list[Path],
        preferred_prior_input: Path | None = None,
        quality_score: float,
        cancel_event: threading.Event | None = None,
    ) -> str:
        from .ai_prior_runner import run_ai_prior_mesh
        from .color_baking import bake_vertex_colors_from_views

        self._publish(job_id, JobStatus.PROCESSING, stage="phase_confidence_routing", progress=41)
        band, route = self._route_from_confidence(float(quality_score))
        if route == "fail":
            raise RuntimeError(
                f"Low reconstruction confidence ({quality_score:.3f}); "
                "capture set is too weak for hybrid reconstruction."
            )

        self._publish(job_id, JobStatus.PROCESSING, stage="phase_prior_generation", progress=52)
        prior = run_ai_prior_mesh(
            job_id=job_id,
            masked_images=masked_paths,
            original_images=original_paths,
            settings=self.runtime_settings,
            work_dir=self.config.root_dir / "data" / "ai_prior_workspace" / job_id,
            preferred_input_image=preferred_prior_input,
        )
        mesh = keep_largest_mesh_component(prior.mesh)
        route_taken = "prior_only"
        quality_reason = f"{band}_confidence_input"

        if route == "hybrid_refine":
            refine_backend = str(getattr(self.runtime_settings, "hybrid_refine_backend", "mapanything")).strip().lower()
            if refine_backend != "none":
                geometry_source = str(getattr(self.runtime_settings, "reconstruction_image_source", "original")).strip().lower()
                reconstruction_inputs = original_paths if geometry_source == "original" else masked_paths
                self._publish(job_id, JobStatus.PROCESSING, stage="phase_prior_refinement", progress=66)
                recon = self.reconstructor.reconstruct(
                    reconstruction_inputs,
                    job_id=job_id,
                    mesh_backend=refine_backend,  # type: ignore[arg-type]
                )
                if len(recon.aligned_points_xyz) >= int(getattr(self.runtime_settings, "hybrid_min_refine_points", 5000)):
                    mesh = self._refine_prior_mesh_with_points(
                        mesh,
                        recon.aligned_points_xyz,
                        strength=float(getattr(self.runtime_settings, "hybrid_refine_strength", 0.2)),
                    )
                    pcd = build_point_cloud(
                        recon.aligned_points_xyz,
                        colors_rgb=recon.aligned_colors_rgb,
                    )
                    if pcd.has_colors():
                        mesh = transfer_vertex_colors_from_point_cloud(mesh, pcd)
                    if bool(getattr(self.runtime_settings, "hybrid_enable_photo_bake", True)):
                        source_remap = self._prepare_texture_source_remap(
                            job_id=job_id,
                            original_paths=original_paths,
                            masked_paths=masked_paths,
                        )
                        views = self._build_photo_views(
                            recon,
                            original_paths=original_paths,
                            masked_paths=masked_paths,
                            source_remap=source_remap,
                        )
                        if views:
                            self._publish(job_id, JobStatus.PROCESSING, stage="photo_vertex_bake", progress=86)
                            bake_vertex_colors_from_views(mesh, views)
                    route_taken = "hybrid_refine"
                    quality_reason = "high_confidence_hybrid_refine"
                else:
                    quality_reason = "insufficient_refine_points_prior_kept"

        if route == "coarse_prior":
            mesh = decimate(mesh, max(10_000, int(self.config.decimation_target_triangles * 0.25)))
            route_taken = "coarse_prior"
            quality_reason = "low_confidence_coarse_prior"

        mesh = keep_largest_mesh_component(mesh)
        mesh = center_and_scale_mesh(mesh)
        mesh = autobalance_vertex_colors(mesh)

        self._publish(job_id, JobStatus.PROCESSING, stage="exporting", progress=94)
        glb_path = self.config.output_dir / f"{job_id}.glb"
        export_glb(
            mesh,
            glb_path,
            compressed=bool(self.runtime_settings.mesh_glb_draco_compression),
        )
        model_url = f"{self.config.cdn_base_url.rstrip('/')}/{job_id}.glb"
        self._write_reconstruction_report(
            job_id,
            {
                "job_id": job_id,
                "reconstruction_confidence": round(float(quality_score), 4),
                "route_taken": route_taken,
                "quality_reason": quality_reason,
                "ai_prior_provider": prior.provider,
                "ai_prior_confidence": round(float(prior.confidence), 4),
                "details": prior.details,
            },
        )
        self._publish(
            job_id,
            JobStatus.COMPLETED,
            stage="completed",
            progress=100,
            model_url=model_url,
            model_format="glb",
        )
        return model_url

    def _write_texture_report(self, job_id: str, payload: dict) -> None:
        out = self.config.root_dir / "uploads" / job_id / "texture_report.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _run_mesh_pipeline(
        self,
        job_id: str,
        masked_paths: list[Path],
        original_paths: list[Path],
        *,
        cancel_event: threading.Event | None = None,
        mesh_backend: Literal["mapanything"] | None = None,
        output_basename: str | None = None,
        publish_completed: bool = True,
    ) -> str:
        stem = output_basename or job_id
        geometry_source = str(getattr(self.runtime_settings, "reconstruction_image_source", "original")).strip().lower()
        reconstruction_inputs = original_paths if geometry_source == "original" else masked_paths
        if len(reconstruction_inputs) < 2:
            raise RuntimeError(
                f"Not enough reconstruction inputs ({len(reconstruction_inputs)}) for source={geometry_source!r}"
            )
        # Phase 2 of the protocol: MapAnything metric reconstruction (+ optional confidence masking there).
        self._publish(job_id, JobStatus.PROCESSING, stage="phase_2_alignment", progress=45)
        reconstruction = self.reconstructor.reconstruct(
            reconstruction_inputs, job_id=job_id, mesh_backend=mesh_backend
        )
        self._raise_if_cancelled(cancel_event)

        # Phase 3 of the protocol: Statistical Outlier Removal on the unified cloud.
        self._publish(job_id, JobStatus.PROCESSING, stage="phase_3_sanitization", progress=65)
        pcd = build_point_cloud(
            reconstruction.aligned_points_xyz,
            colors_rgb=reconstruction.aligned_colors_rgb,
        )
        clean_pcd = remove_statistical_outliers(
            pcd,
            nb_neighbors=self.config.nb_neighbors,
            std_ratio=self.config.std_ratio,
        )
        self._raise_if_cancelled(cancel_event)

        # Mesh-only finishing steps (not part of the GS protocol; .glb path).
        self._publish(job_id, JobStatus.PROCESSING, stage="meshing", progress=75)
        mesh = poisson_mesh(
            clean_pcd,
            depth=self.config.poisson_depth,
            density_quantile=self.config.poisson_density_quantile,
        )
        self._raise_if_cancelled(cancel_event)
        mesh = decimate(mesh, self.config.decimation_target_triangles)
        self._raise_if_cancelled(cancel_event)
        self._publish(job_id, JobStatus.PROCESSING, stage="mesh_cleanup", progress=79)
        mesh = keep_largest_mesh_component(mesh)
        self._raise_if_cancelled(cancel_event)

        # Optional semantic shape correction (classify object, align class template, blend correction).
        try:
            from .shape_prior import apply_shape_prior_correction

            self._publish(job_id, JobStatus.PROCESSING, stage="shape_classification", progress=81)
            mesh, prior_report = apply_shape_prior_correction(mesh, list(original_paths), self.runtime_settings)
            if prior_report.applied:
                self._publish(job_id, JobStatus.PROCESSING, stage="shape_template_correction", progress=84)
                _log.info(
                    "Shape prior applied job=%s label=%s conf=%.3f rmse=%s template=%s",
                    job_id,
                    prior_report.label,
                    prior_report.confidence,
                    (
                        f"{prior_report.alignment_rmse:.4f}"
                        if prior_report.alignment_rmse is not None
                        else "n/a"
                    ),
                    prior_report.template_path,
                )
            else:
                _log.info(
                    "Shape prior skipped job=%s reason=%s label=%s conf=%.3f",
                    job_id,
                    prior_report.reason,
                    prior_report.label,
                    prior_report.confidence,
                )
        except Exception as exc:
            _log.warning("shape prior correction failed job=%s: %s", job_id, exc)

        # Make sure point-cloud colours actually end up on the GLB. Open3D's Poisson +
        # decimation don't reliably propagate vertex colors across versions, so
        # we always transfer them from the cleaned colored cloud at the end.
        if clean_pcd.has_colors():
            self._publish(job_id, JobStatus.PROCESSING, stage="vertex_color_transfer", progress=85)
            mesh = transfer_vertex_colors_from_point_cloud(mesh, clean_pcd)

        if self.runtime_settings.mesh_photo_vertex_bake:
            photo_views: list[CameraView] = []
            region_meta: dict[str, dict[str, object]] = {}
            try:
                source_remap, region_meta = self._prepare_texture_source_remap(
                    job_id=job_id,
                    original_paths=original_paths,
                    masked_paths=masked_paths,
                )
                photo_views = self._build_photo_views(
                    reconstruction,
                    original_paths=original_paths,
                    masked_paths=masked_paths,
                    source_remap=source_remap,
                )
            except Exception:
                photo_views = []

            if photo_views:
                try:
                    from .color_baking import bake_vertex_colors_from_views

                    projection_min_conf = float(
                        getattr(self.runtime_settings, "region_projection_min_confidence", 0.60)
                    )
                    for view in photo_views:
                        key = str(Path(view.image_path).resolve())
                        meta = region_meta.get(key)
                        if meta is None:
                            meta = region_meta.get(Path(view.image_path).name)
                        if meta:
                            view.surface_region_confidence = float(meta.get("confidence", 1.0))
                            lh = meta.get("label_hint")
                            view.surface_region_label = str(lh) if lh else None
                        else:
                            view.surface_region_confidence = 1.0
                    self._publish(
                        job_id,
                        JobStatus.PROCESSING,
                        stage="phase_surface_region_projection",
                        progress=86,
                    )
                    baked, bake_diag = bake_vertex_colors_from_views(
                        mesh,
                        photo_views,
                        blend_weight=float(getattr(self.runtime_settings, "region_projection_blend_weight", 0.65)),
                        min_confidence=projection_min_conf,
                        seam_smoothing=float(getattr(self.runtime_settings, "region_projection_seam_smoothing", 0.55)),
                    )
                    texture_route_taken = "mapanything"
                    texture_quality_reason = "photo_bake_missing_views"
                    if baked:
                        self._publish(
                            job_id,
                            JobStatus.PROCESSING,
                            stage="phase_surface_region_blend",
                            progress=89,
                        )
                        regions_projected = int(
                            (bake_diag.region_projection_coverage or {}).get("regions_projected", 0)
                        )
                        mesh_cov = float(
                            (bake_diag.region_projection_coverage or {}).get("mesh_area_ratio", 0.0)
                        )
                        region_enabled = bool(getattr(self.runtime_settings, "surface_region_texture_enabled", False))
                        if region_enabled and regions_projected > 0:
                            if mesh_cov >= 0.55:
                                texture_route_taken = "surface_region"
                                texture_quality_reason = "region_projection_coverage_ok"
                            else:
                                texture_route_taken = "surface_region+mapanything_fill"
                                texture_quality_reason = "low_region_projection_coverage_hybrid_fill"
                        else:
                            texture_route_taken = "mapanything"
                            texture_quality_reason = "surface_region_disabled_or_low_confidence"
                        self._write_texture_report(
                            job_id,
                            {
                                "job_id": job_id,
                                "dominant_surface_regions_detected": bake_diag.dominant_surface_regions_detected,
                                "per_view_region_confidence": bake_diag.per_view_region_confidence,
                                "region_projection_coverage": bake_diag.region_projection_coverage,
                                "texture_route_taken": texture_route_taken,
                                "texture_quality_reason": texture_quality_reason,
                                "projection_min_confidence": projection_min_conf,
                            },
                        )
                    else:
                        self._write_texture_report(
                            job_id,
                            {
                                "job_id": job_id,
                                "dominant_surface_regions_detected": 0,
                                "per_view_region_confidence": {},
                                "region_projection_coverage": {"mesh_area_ratio": 0.0, "regions_projected": 0},
                                "texture_route_taken": "mapanything",
                                "texture_quality_reason": "region_projection_failed_fallback_mapanything",
                                "projection_min_confidence": projection_min_conf,
                            },
                        )
                except Exception as exc:
                    _log.warning(
                        "bake_vertex_colors_from_views failed job=%s (mesh may look flat/dark): %s",
                        job_id,
                        exc,
                    )
                    self._write_texture_report(
                        job_id,
                        {
                            "job_id": job_id,
                            "dominant_surface_regions_detected": 0,
                            "per_view_region_confidence": {},
                            "region_projection_coverage": {"mesh_area_ratio": 0.0, "regions_projected": 0},
                            "texture_route_taken": "mapanything",
                            "texture_quality_reason": "projection_exception_fallback_mapanything",
                        },
                    )
            else:
                self._write_texture_report(
                    job_id,
                    {
                        "job_id": job_id,
                        "dominant_surface_regions_detected": 0,
                        "per_view_region_confidence": {},
                        "region_projection_coverage": {"mesh_area_ratio": 0.0, "regions_projected": 0},
                        "texture_route_taken": "mapanything",
                        "texture_quality_reason": "no_photo_views_fallback_mapanything",
                    },
                )

        self._publish(job_id, JobStatus.PROCESSING, stage="color_autobalance", progress=90)
        mesh = autobalance_vertex_colors(mesh)
        mesh = center_and_scale_mesh(mesh)

        self._publish(job_id, JobStatus.PROCESSING, stage="exporting", progress=94)
        glb_path = self.config.output_dir / f"{stem}.glb"
        export_glb(
            mesh,
            glb_path,
            compressed=bool(self.runtime_settings.mesh_glb_draco_compression),
        )
        if not glb_path.is_file() or glb_path.stat().st_size < 256:
            raise RuntimeError(f"Export produced no usable GLB at {glb_path}")

        model_url = f"{self.config.cdn_base_url.rstrip('/')}/{stem}.glb"
        if publish_completed:
            self._publish(
                job_id,
                JobStatus.COMPLETED,
                stage="completed",
                progress=100,
                model_url=model_url,
                model_format="glb",
            )
        return model_url

    def _run_gaussian_splatting(
        self,
        job_id: str,
        masked_paths: list[Path],
        original_paths: list[Path],
        *,
        cancel_event: threading.Event | None = None,
    ) -> str:
        from .gaussian_splatting_runner import run_gaussian_splatting

        self._raise_if_cancelled(cancel_event)
        ply_path = self.config.output_dir / f"{job_id}.ply"
        work_dir = self.config.root_dir / "data" / "gs_workspace" / job_id

        def _on_progress(stage: str, progress: int) -> None:
            self._publish(job_id, JobStatus.PROCESSING, stage=stage, progress=progress)

        run_gaussian_splatting(
            job_id=job_id,
            masked_images=masked_paths,
            work_dir=work_dir,
            output_ply=ply_path,
            settings=self.runtime_settings,
            progress_callback=_on_progress,
            original_training_images=list(original_paths),
        )

        if not ply_path.is_file() or ply_path.stat().st_size < 256:
            raise RuntimeError(f"Gaussian Splatting produced no usable PLY at {ply_path}")

        self._publish(job_id, JobStatus.PROCESSING, stage="exporting", progress=95)
        model_url = f"{self.config.cdn_base_url.rstrip('/')}/{job_id}.ply"
        self._publish(
            job_id,
            JobStatus.COMPLETED,
            stage="completed",
            progress=100,
            model_url=model_url,
            model_format="ply",
        )
        return model_url

    @staticmethod
    def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise JobCancelled()

    def _run_segmentation_isolated_subprocess(self, job_id: str, image_paths: list[Path]) -> list[Path]:
        from .cuda_memory import ensure_cuda_allocator_env

        ensure_cuda_allocator_env()
        scratch = self.config.root_dir / "data" / "phase_scratch" / job_id
        scratch.mkdir(parents=True, exist_ok=True)
        settings_json = scratch / "sam_settings.json"
        paths_json = scratch / "sam_paths.json"
        settings_json.write_text(self.runtime_settings.model_dump_json(), encoding="utf-8")
        paths_json.write_text(json.dumps([str(p.resolve()) for p in image_paths]), encoding="utf-8")

        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "worker.gpu_phase_sam",
                "--job-id",
                job_id,
                "--settings-json",
                str(settings_json),
                "--paths-json",
                str(paths_json),
                "--root-dir",
                str(self.config.root_dir.resolve()),
            ],
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
            env=os.environ.copy(),
            check=False,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-40:]
            raise RuntimeError(
                "SAM segmentation subprocess failed.\nCommand: worker.gpu_phase_sam\n" + "\n".join(tail)
            )
        purge_torch_cuda()

        masked_paths: list[Path] = []
        for idx, image_path in enumerate(image_paths):
            suffix = image_path.suffix or ".png"
            out = self.config.masked_dir / job_id / f"masked_{idx:03d}{suffix}"
            if not out.is_file():
                raise RuntimeError(f"SAM subprocess did not write expected masked image: {out}")
            masked_paths.append(out)
        return masked_paths

    def _run_segmentation(
        self,
        job_id: str,
        image_paths: list[Path],
        *,
        cancel_event: threading.Event | None = None,
    ) -> list[Path]:
        if not image_paths:
            raise ValueError("No input images provided")

        total = max(1, len(image_paths))

        isolate = False
        try:
            import torch

            isolate = (
                bool(torch.cuda.is_available())
                and effective_gpu_isolate_phases(self.runtime_settings)
                and not self.runtime_settings.allow_placeholder_pipeline
                and cancel_event is None
            )
        except Exception:
            isolate = False

        if isolate:
            masked_paths = self._run_segmentation_isolated_subprocess(job_id, image_paths)
            masked_paths = self._filter_segmentation_outliers(job_id, masked_paths)
            total_n = max(1, len(masked_paths))
            self._publish(
                job_id,
                JobStatus.PROCESSING,
                stage=f"phase_1_segmentation ({total_n}/{total_n})",
                progress=40,
            )
            return masked_paths

        masked_paths: list[Path] = []
        save_masks = bool(self.runtime_settings.save_raw_masks)
        masks_root = self.config.masks_dir / job_id if save_masks else None
        # Segmentation occupies the 10..40 progress band.
        SEG_START, SEG_END = 10, 40
        for idx, image_path in enumerate(image_paths):
            self._raise_if_cancelled(cancel_event)
            suffix = image_path.suffix or ".png"
            output_path = self.config.masked_dir / job_id / f"masked_{idx:03d}{suffix}"
            mask_path = (masks_root / f"mask_{idx:03d}.png") if masks_root is not None else None
            masked_path = self.segmenter.segment_file(
                image_path,
                output_path,
                mask_output_path=mask_path,
            )
            masked_paths.append(masked_path)
            pct = SEG_START + int((SEG_END - SEG_START) * (idx + 1) / total)
            self._publish(
                job_id,
                JobStatus.PROCESSING,
                stage=f"phase_1_segmentation ({idx + 1}/{total})",
                progress=pct,
            )
        return self._filter_segmentation_outliers(job_id, masked_paths)

    def _filter_segmentation_outliers(self, job_id: str, masked_paths: list[Path]) -> list[Path]:
        if not bool(getattr(self.runtime_settings, "isolation_filter_outlier_views", True)):
            return masked_paths
        if len(masked_paths) < 4:
            return masked_paths

        metrics: list[tuple[Path, float, float]] = []
        for path in masked_paths:
            m = self._masked_view_metrics(path)
            if m is not None:
                metrics.append((path, m[0], m[1]))
        if len(metrics) < 4:
            return masked_paths

        areas = np.asarray([m[1] for m in metrics], dtype=np.float64)
        area_med = float(np.median(areas))
        mad = float(np.median(np.abs(areas - area_med)))
        area_scale = max(1e-6, 1.4826 * mad)
        mad_mult = float(getattr(self.runtime_settings, "isolation_outlier_area_mad_scale", 3.2))
        max_center_dist = float(getattr(self.runtime_settings, "isolation_outlier_center_distance", 0.22))

        kept: list[Path] = []
        dropped = 0
        for p, area_ratio, cdist in metrics:
            z = abs(area_ratio - area_med) / area_scale
            center_ok = cdist <= max_center_dist
            area_ok = z <= mad_mult
            if area_ok and center_ok:
                kept.append(p)
            else:
                dropped += 1

        # Safety: never collapse to too few views.
        min_keep = 3 if len(masked_paths) < 8 else 4
        if len(kept) < min_keep:
            return masked_paths
        if dropped > 0:
            _log.info("Segmentation outlier filter dropped %s/%s views for job=%s", dropped, len(masked_paths), job_id)
        return kept

    @staticmethod
    def _masked_view_metrics(path: Path) -> tuple[float, float] | None:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        fg = np.any(img > 8, axis=2)
        if not np.any(fg):
            return None
        h, w = fg.shape[:2]
        ys, xs = np.where(fg)
        area_ratio = float(fg.mean())
        cx = float(xs.mean()) / max(1.0, float(w - 1))
        cy = float(ys.mean()) / max(1.0, float(h - 1))
        center_dist = float(np.hypot(cx - 0.5, cy - 0.5))
        return area_ratio, center_dist

    def _publish(
        self,
        job_id: str,
        status: JobStatus,
        *,
        stage: str | None = None,
        progress: int | None = None,
        model_url: str | None = None,
        model_format: str | None = None,
        error: str | None = None,
    ) -> None:
        self.job_repo.set_status(
            job_id,
            status,
            stage=stage,
            progress=progress,
            model_url=model_url,
            model_format=model_format,
            error=error,
        )
        self.notifier.notify_job_update(
            job_id,
            status,
            stage=stage,
            progress=progress,
            model_url=model_url,
            model_format=model_format,
            error=error,
        )

