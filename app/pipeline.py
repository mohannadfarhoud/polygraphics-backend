from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

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
from .runtime_settings import RuntimeSettings, effective_reconstruction_backend, gaussian_splatting_skipped_via_env
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

            image_paths = downscale_job_images_if_needed(
                job_id,
                image_paths,
                self.config.root_dir / "data" / "job_inputs",
                int(self.runtime_settings.max_input_image_side),
            )
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
            try:
                self.segmenter.release_gpu_memory()
            except Exception:
                pass
            purge_torch_cuda()
            self._raise_if_cancelled(cancel_event)

            if effective_reconstruction_backend(self.runtime_settings) == "gaussian_splatting":
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

            model_url = self._run_mesh_pipeline(
                job_id, masked_paths, originals_for_mesh, cancel_event=cancel_event
            )
            return model_url
        except JobCancelled:
            raise
        except Exception as exc:
            self._publish(job_id, JobStatus.FAILED, error=str(exc))
            raise

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

        # Make sure point-cloud colours actually end up on the GLB. Open3D's Poisson +
        # decimation don't reliably propagate vertex colors across versions, so
        # we always transfer them from the cleaned colored cloud at the end.
        if clean_pcd.has_colors():
            self._publish(job_id, JobStatus.PROCESSING, stage="vertex_color_transfer", progress=77)
            mesh = transfer_vertex_colors_from_point_cloud(mesh, clean_pcd)

        if self.runtime_settings.mesh_photo_vertex_bake:
            photo_views: list[CameraView] = []
            try:
                cams = list(getattr(reconstruction, "cameras", []) or [])
                if cams:
                    masked_to_original: dict[str, Path] = {}
                    original_to_masked: dict[str, Path] = {}
                    for masked, original in zip(masked_paths, original_paths):
                        masked_to_original[str(masked)] = original
                        masked_to_original[masked.name] = original
                        original_to_masked[str(original)] = masked
                        original_to_masked[original.name] = masked

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
                        photo_views.append(
                            CameraView(
                                image_path=sample_path,
                                image_size=cam.image_size,
                                K=cam.K,
                                w2c=cam.w2c,
                            )
                        )
            except Exception:
                photo_views = []

            if photo_views:
                try:
                    from .color_baking import bake_vertex_colors_from_views

                    self._publish(job_id, JobStatus.PROCESSING, stage="photo_vertex_bake", progress=86)
                    bake_vertex_colors_from_views(mesh, photo_views)
                except Exception as exc:
                    _log.warning(
                        "bake_vertex_colors_from_views failed job=%s (mesh may look flat/dark): %s",
                        job_id,
                        exc,
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

