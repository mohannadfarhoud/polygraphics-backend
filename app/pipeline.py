from __future__ import annotations

import threading
from pathlib import Path

from .config import PipelineConfig
from .interfaces import JobRepository, JobStatus, NoopJobRepository, NoopWebSocketNotifier, WebSocketNotifier
from .meshing import decimate, export_glb, poisson_mesh
from .pipeline_ready import assert_pipeline_ready
from .point_cloud import build_point_cloud, remove_statistical_outliers
from .reconstruction import Dust3RReconstructor
from .runtime_settings import RuntimeSettings
from .segmentation import SamSegmenter


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
                model_url = self._run_gaussian_splatting(job_id, masked_paths, cancel_event=cancel_event)
                return model_url

            model_url = self._run_mesh_pipeline(job_id, masked_paths, cancel_event=cancel_event)
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
        *,
        cancel_event: threading.Event | None = None,
    ) -> str:
        self._publish(job_id, JobStatus.PROCESSING, stage="reconstructing", progress=40)
        reconstruction = self.reconstructor.reconstruct(masked_paths)
        self._raise_if_cancelled(cancel_event)

        self._publish(job_id, JobStatus.PROCESSING, stage="cleaning", progress=70)
        pcd = build_point_cloud(reconstruction.aligned_points_xyz)
        clean_pcd = remove_statistical_outliers(
            pcd,
            nb_neighbors=self.config.nb_neighbors,
            std_ratio=self.config.std_ratio,
        )
        self._raise_if_cancelled(cancel_event)

        self._publish(job_id, JobStatus.PROCESSING, stage="meshing", progress=80)
        mesh = poisson_mesh(
            clean_pcd,
            depth=self.config.poisson_depth,
            density_quantile=self.config.poisson_density_quantile,
        )
        self._raise_if_cancelled(cancel_event)
        mesh = decimate(mesh, self.config.decimation_target_triangles)
        self._raise_if_cancelled(cancel_event)

        self._publish(job_id, JobStatus.PROCESSING, stage="exporting", progress=95)
        glb_path = self.config.output_dir / f"{job_id}.glb"
        export_glb(mesh, glb_path)
        if not glb_path.is_file() or glb_path.stat().st_size < 256:
            raise RuntimeError(f"Export produced no usable GLB at {glb_path}")

        model_url = f"{self.config.cdn_base_url.rstrip('/')}/{job_id}.glb"
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
        self._publish(job_id, JobStatus.PROCESSING, stage="training_gs", progress=50)
        ply_path = self.config.output_dir / f"{job_id}.ply"
        work_dir = self.config.root_dir / "data" / "gs_workspace" / job_id

        run_gaussian_splatting(
            job_id=job_id,
            masked_images=masked_paths,
            work_dir=work_dir,
            output_ply=ply_path,
            settings=self.runtime_settings,
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
        # Segmentation occupies the 10..40 progress band.
        SEG_START, SEG_END = 10, 40
        for idx, image_path in enumerate(image_paths):
            self._raise_if_cancelled(cancel_event)
            suffix = image_path.suffix or ".png"
            output_path = self.config.masked_dir / job_id / f"masked_{idx:03d}{suffix}"
            masked_path = self.segmenter.segment_file(image_path, output_path)
            masked_paths.append(masked_path)
            pct = SEG_START + int((SEG_END - SEG_START) * (idx + 1) / total)
            self._publish(
                job_id,
                JobStatus.PROCESSING,
                stage=f"segmenting ({idx + 1}/{total})",
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

