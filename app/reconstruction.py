from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .runtime_settings import RuntimeSettings


@dataclass
class ReconstructionResult:
    aligned_points_xyz: np.ndarray


class Dust3RReconstructor:
    """DUSt3R reconstruction; demo cloud only when allow_placeholder_pipeline is True."""

    def __init__(self, settings: RuntimeSettings | None = None) -> None:
        self.settings = settings

    def reconstruct(self, masked_images: list[Path]) -> ReconstructionResult:
        if len(masked_images) < 2:
            raise ValueError("Need at least 2 masked images")

        if self.settings is None:
            raise RuntimeError("Pipeline misconfigured: runtime settings missing.")

        if self.settings.allow_placeholder_pipeline:
            return ReconstructionResult(aligned_points_xyz=self._placeholder_cloud(masked_images))

        if self.settings.reconstruction_backend == "colmap":
            raise RuntimeError(
                "COLMAP backend is not implemented in this service yet. "
                "Use reconstruction_backend=dust3r with a DUSt3R checkpoint, "
                "or set allow_placeholder_pipeline=true for demos only."
            )

        from .dust3r_runner import run_dust3r_point_cloud

        cloud = run_dust3r_point_cloud(masked_images, self.settings)
        return ReconstructionResult(aligned_points_xyz=cloud)

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
