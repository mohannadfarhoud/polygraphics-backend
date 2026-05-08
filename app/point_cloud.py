from __future__ import annotations

import numpy as np
import open3d as o3d


def build_point_cloud(
    points_xyz: np.ndarray,
    colors_rgb: np.ndarray | None = None,
) -> o3d.geometry.PointCloud:
    """Build an Open3D point cloud, optionally with per-point colors.

    ``colors_rgb`` is expected as Nx3 ``uint8`` (0-255) or already-normalized
    ``float`` in [0, 1]. Open3D stores colors as float64 in [0, 1].
    """
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must be Nx3")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)

    if colors_rgb is not None and len(colors_rgb) == len(points_xyz):
        c = np.asarray(colors_rgb)
        if c.ndim == 2 and c.shape[1] >= 3:
            rgb = c[:, :3].astype(np.float64, copy=False)
            if rgb.size and float(rgb.max()) > 1.0:
                rgb = rgb / 255.0
            pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def remove_statistical_outliers(
    pcd: o3d.geometry.PointCloud,
    *,
    nb_neighbors: int,
    std_ratio: float,
) -> o3d.geometry.PointCloud:
    _, inlier_indices = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    if not inlier_indices:
        raise RuntimeError("Outlier removal dropped all points")
    return pcd.select_by_index(inlier_indices)

