"""Shared sparse multi-view reconstruction payload for mesh export and GS COLMAP-text bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class MultiviewMetricScene:
    """Merged metric reconstruction: point cloud plus per-view pinhole geometry.

    Compatible with ``colmap_bridge.write_colmap_text`` (COLMAP *text* layout for Gaussian Splatting).
    """

    points: np.ndarray  # (N, 3) float32, world frame
    colors: np.ndarray  # (N, 3) uint8 RGB
    image_paths: list[Path] = field(default_factory=list)
    image_sizes: list[tuple[int, int]] = field(default_factory=list)  # (W, H) per image
    intrinsics: list[np.ndarray] = field(default_factory=list)  # 3x3 K per image
    poses_w2c: list[np.ndarray] = field(default_factory=list)  # 4x4 world→camera
    poses_c2w: list[np.ndarray] = field(default_factory=list)  # 4x4 camera→world
