from __future__ import annotations

import numpy as np
import open3d as o3d


def build_point_cloud(points_xyz: np.ndarray) -> o3d.geometry.PointCloud:
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError("points_xyz must be Nx3")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
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

