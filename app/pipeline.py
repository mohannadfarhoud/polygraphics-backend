from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Literal

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
from .pipeline_ready import assert_pipeline_ready
from .point_cloud import build_point_cloud, remove_statistical_outliers
from .reconstruction import Dust3RReconstructor
from .runtime_settings import RuntimeSettings
from .segmentation import SamSegmenter

_log = logging.getLogger(__name__)


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
            masked_paths = self._run_segmentation(job_id, image_paths, cancel_event=cancel_event)
            self._raise_if_cancelled(cancel_event)

            if self.runtime_settings.reconstruction_backend == "gaussian_splatting":
                if self.runtime_settings.compare_mesh_dust3r_colmap_with_gs:
                    self._publish(
                        job_id,
                        JobStatus.PROCESSING,
                        stage="compare_mesh_dust3r",
                        progress=41,
                    )
                    self._run_mesh_pipeline(
                        job_id,
                        masked_paths,
                        image_paths,
                        cancel_event=cancel_event,
                        mesh_backend="dust3r",
                        output_basename=f"{job_id}_compare_dust3r",
                        publish_completed=False,
                    )
                    self._raise_if_cancelled(cancel_event)
                    self._publish(
                        job_id,
                        JobStatus.PROCESSING,
                        stage="compare_mesh_colmap",
                        progress=44,
                    )
                    self._run_mesh_pipeline(
                        job_id,
                        masked_paths,
                        image_paths,
                        cancel_event=cancel_event,
                        mesh_backend="colmap",
                        output_basename=f"{job_id}_compare_colmap",
                        publish_completed=False,
                    )
                    self._raise_if_cancelled(cancel_event)
                model_url = self._run_gaussian_splatting(job_id, masked_paths, cancel_event=cancel_event)
                return model_url

            model_url = self._run_mesh_pipeline(
                job_id, masked_paths, image_paths, cancel_event=cancel_event
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
        mesh_backend: Literal["dust3r", "colmap"] | None = None,
        output_basename: str | None = None,
        publish_completed: bool = True,
    ) -> str:
        stem = output_basename or job_id
        # Phase 2 of the protocol: DUSt3R/COLMAP reconstruction (also applies the
        # confidence filter from Phase 3 before merging per-view clouds for DUSt3R).
        self._publish(job_id, JobStatus.PROCESSING, stage="phase_2_alignment", progress=45)
        reconstruction = self.reconstructor.reconstruct(
            masked_paths, job_id=job_id, mesh_backend=mesh_backend
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

        self._publish(job_id, JobStatus.PROCESSING, stage="color_autobalance", progress=90)
        mesh = autobalance_vertex_colors(mesh)
        mesh = center_and_scale_mesh(mesh)

        self._publish(job_id, JobStatus.PROCESSING, stage="exporting", progress=94)
        glb_path = self.config.output_dir / f"{stem}.glb"
        export_glb(mesh, glb_path)
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
        return masked_paths

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

