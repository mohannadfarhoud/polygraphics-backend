from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .runtime_settings import RuntimeSettings


@dataclass
class ReconstructionResult:
    aligned_points_xyz: np.ndarray  # (N, 3) float32 in a unified frame
    aligned_colors_rgb: np.ndarray | None = None  # (N, 3) uint8 RGB; same length as points


class Dust3RReconstructor:
    """Runs the active mesh-path backend (DUSt3R / COLMAP / auto).

    The class name is historical; today it dispatches on
    ``settings.reconstruction_backend`` (or ``auto`` based on image count).
    """

    def __init__(self, settings: RuntimeSettings | None = None) -> None:
        self.settings = settings

    def reconstruct(
        self,
        masked_images: list[Path],
        *,
        job_id: str | None = None,
    ) -> ReconstructionResult:
        if len(masked_images) < 2:
            raise ValueError("Need at least 2 masked images")

        if self.settings is None:
            raise RuntimeError("Pipeline misconfigured: runtime settings missing.")

        if self.settings.allow_placeholder_pipeline:
            return ReconstructionResult(
                aligned_points_xyz=self._placeholder_cloud(masked_images),
                aligned_colors_rgb=None,
            )

        backend = self.resolve_backend(len(masked_images))

        if backend == "dust3r":
            from .dust3r_runner import run_dust3r_scene

            scene = run_dust3r_scene(masked_images, self.settings)
            return ReconstructionResult(
                aligned_points_xyz=scene.points,
                aligned_colors_rgb=scene.colors,
            )

        if backend == "colmap":
            from .colmap_runner import run_colmap_sparse

            workspace = self._colmap_workspace(masked_images, job_id)
            points, colors = run_colmap_sparse(masked_images, self.settings, workspace=workspace)
            return ReconstructionResult(
                aligned_points_xyz=points,
                aligned_colors_rgb=colors,
            )

        raise RuntimeError(
            f"Unsupported mesh backend {backend!r}. "
            "Use 'auto', 'dust3r', or 'colmap' (or set reconstruction_backend='gaussian_splatting' for the .ply path)."
        )

    def resolve_backend(self, n_images: int) -> str:
        """Return the concrete backend ('dust3r' or 'colmap') given current settings."""
        if self.settings is None:
            return "dust3r"
        backend = self.settings.reconstruction_backend
        if backend != "auto":
            return backend
        threshold = int(self.settings.auto_dust3r_max_images)
        return "dust3r" if n_images < threshold else "colmap"

    @staticmethod
    def _colmap_workspace(masked_images: list[Path], job_id: str | None) -> Path:
        if job_id:
            return Path("data") / "colmap_workspace" / job_id
        # Derive a stable workspace from the masked dir layout
        # (masked/<job_id>/masked_NNN.<ext>) so we don't pile up temp dirs.
        first_parent = masked_images[0].parent
        return first_parent.parent.parent / "data" / "colmap_workspace" / first_parent.name

    def _placeholder_cloud(self, masked_images: list[Path]) -> np.ndarray:
        point_blocks: list[np.ndarray] = []
        for idx, img_path in enumerate(masked_images):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Unable to read masked image: {img_path}")
            ys, xs = np.where(img > 0)
            if xs.size == 0:
                continue
            z = np.full(xs.shape, idx, dtype=np.float32)
            points = np.column_stack([xs.astype(np.float32), ys.astype(np.float32), z])
            point_blocks.append(points)

        if not point_blocks:
            raise RuntimeError("No foreground points detected in masked images")

        cloud = np.vstack(point_blocks)
        cloud -= cloud.mean(axis=0, keepdims=True)
        scale = np.linalg.norm(cloud, axis=1).max()
        if scale > 0:
            cloud /= scale
        return cloud.astype(np.float32)
