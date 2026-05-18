from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from .color_baking import CameraView
from .runtime_settings import RuntimeSettings, effective_reconstruction_backend


@dataclass
class ReconstructionResult:
    aligned_points_xyz: np.ndarray  # (N, 3) float32 in a unified frame
    aligned_colors_rgb: np.ndarray | None = None  # (N, 3) uint8 RGB; same length as points
    cameras: list[CameraView] = field(default_factory=list)
    original_image_paths: list[Path] = field(default_factory=list)


class Dust3RReconstructor:
    """Historical name — runs metric mesh reconstruction via MapAnything (``mapanything_runner``).

    ``gaussian_splatting`` skips this class for the main GS path (scene workspace uses MapAnything directly).
    """

    def __init__(self, settings: RuntimeSettings | None = None) -> None:
        self.settings = settings

    def reconstruct(
        self,
        masked_images: list[Path],
        *,
        job_id: str | None = None,
        mesh_backend: Literal["mapanything", "dust3r"] | None = None,
    ) -> ReconstructionResult:
        del job_id  # Workspaces are ephemeral; GS path uses its own scene dir.
        if len(masked_images) < 2:
            raise ValueError("Need at least 2 masked images")

        if self.settings is None:
            raise RuntimeError("Pipeline misconfigured: runtime settings missing.")

        if self.settings.allow_placeholder_pipeline:
            return ReconstructionResult(
                aligned_points_xyz=self._placeholder_cloud(masked_images),
                aligned_colors_rgb=None,
            )

        backend = (
            mesh_backend
            if mesh_backend is not None
            else effective_reconstruction_backend(self.settings)
        )

        if backend == "gaussian_splatting":
            raise RuntimeError(
                "reconstruct() does not run Gaussian Splatting; set reconstruction_backend to mapanything for mesh,"
                " or gaussian_splatting for .ply exports."
            )

        if backend == "mapanything":
            return self._mapanything_reconstruct(masked_images, self.settings)
        if backend == "dust3r":
            return self._dust3r_reconstruct(masked_images, self.settings)

        raise RuntimeError(f"Unsupported mesh backend {backend!r}.")

    def resolve_backend(self, n_images: int) -> str:
        _ = n_images
        if self.settings is None:
            return "mapanything"
        if effective_reconstruction_backend(self.settings) == "gaussian_splatting":
            return "gaussian_splatting"
        return str(effective_reconstruction_backend(self.settings))

    def _mapanything_reconstruct(self, masked_images: list[Path], settings: RuntimeSettings) -> ReconstructionResult:
        from .mapanything_runner import run_mapanything_scene

        scene = run_mapanything_scene(masked_images, settings)
        cameras: list[CameraView] = []
        for i, masked_path in enumerate(scene.image_paths):
            if i >= len(scene.image_sizes) or i >= len(scene.intrinsics) or i >= len(scene.poses_w2c):
                break
            cameras.append(
                CameraView(
                    image_path=masked_path,
                    image_size=scene.image_sizes[i],
                    K=np.asarray(scene.intrinsics[i], dtype=np.float64),
                    w2c=np.asarray(scene.poses_w2c[i], dtype=np.float64),
                )
            )
        return ReconstructionResult(
            aligned_points_xyz=scene.points,
            aligned_colors_rgb=scene.colors,
            cameras=cameras,
        )

    def _dust3r_reconstruct(self, masked_images: list[Path], settings: RuntimeSettings) -> ReconstructionResult:
        from .dust3r_runner import run_dust3r_scene

        scene = run_dust3r_scene(masked_images, settings)
        cameras: list[CameraView] = []
        for i, masked_path in enumerate(scene.image_paths):
            if i >= len(scene.image_sizes) or i >= len(scene.intrinsics) or i >= len(scene.poses_w2c):
                break
            cameras.append(
                CameraView(
                    image_path=masked_path,
                    image_size=scene.image_sizes[i],
                    K=np.asarray(scene.intrinsics[i], dtype=np.float64),
                    w2c=np.asarray(scene.poses_w2c[i], dtype=np.float64),
                )
            )
        return ReconstructionResult(
            aligned_points_xyz=scene.points,
            aligned_colors_rgb=scene.colors,
            cameras=cameras,
        )

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
