"""Bake mesh-vertex colors from original photos using camera projections.

Given a list of camera views ``(image_path, image_size, K, w2c)`` aligned to the
mesh's world frame, project each mesh vertex into every image, sample the colour,
and average across visible views. This produces photo-true vertex colours even
when the upstream point-cloud colours were stripped, normalised, or otherwise
came out near-black.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


@dataclass
class CameraView:
    image_path: Path  # original (preferably unmasked) image on disk
    image_size: tuple[int, int]  # (W, H) the K below was estimated for
    K: np.ndarray  # 3x3 intrinsics matrix (float64)
    w2c: np.ndarray  # 4x4 world->camera transform (float64)


def _read_image_at(path: Path, size: tuple[int, int]) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    W, H = int(size[0]), int(size[1])
    if (img.shape[1], img.shape[0]) != (W, H):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def bake_vertex_colors_from_views(
    mesh: o3d.geometry.TriangleMesh,
    views: list[CameraView],
    *,
    skip_dark_threshold: int = 8,
) -> bool:
    """Sample colours per-vertex from ``views`` and overwrite ``mesh.vertex_colors``.

    Pixels darker than ``skip_dark_threshold`` (sum of channels) are skipped so
    masked-out (black) backgrounds don't pollute the average. Returns ``True``
    when at least one vertex received a sample, ``False`` otherwise.
    """
    verts = np.asarray(mesh.vertices)
    if verts.size == 0 or not views:
        return False

    images: list[np.ndarray | None] = [_read_image_at(v.image_path, v.image_size) for v in views]

    n = verts.shape[0]
    color_sum = np.zeros((n, 3), dtype=np.float64)
    color_cnt = np.zeros(n, dtype=np.int32)
    homog = np.concatenate([verts, np.ones((n, 1), dtype=np.float64)], axis=1)  # Nx4

    for v, img in zip(views, images):
        if img is None:
            continue
        W, H = int(v.image_size[0]), int(v.image_size[1])

        pcam = (np.asarray(v.w2c, dtype=np.float64) @ homog.T).T  # Nx4
        z = pcam[:, 2]
        valid_z = z > 1e-6
        if not valid_z.any():
            continue

        K = np.asarray(v.K, dtype=np.float64)
        x = pcam[valid_z, 0] / z[valid_z]
        y = pcam[valid_z, 1] / z[valid_z]
        u = K[0, 0] * x + K[0, 2]
        vp = K[1, 1] * y + K[1, 2]

        ui = np.floor(u + 0.5).astype(np.int32)
        vi = np.floor(vp + 0.5).astype(np.int32)
        in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
        if not in_bounds.any():
            continue

        global_idx = np.where(valid_z)[0][in_bounds]
        ui_b = ui[in_bounds]
        vi_b = vi[in_bounds]

        rgb = img[vi_b, ui_b, :]  # uint8 RGB
        sums = rgb.astype(np.int32).sum(axis=1)
        non_dark = sums > int(skip_dark_threshold)
        if not non_dark.any():
            continue

        keep_idx = global_idx[non_dark]
        keep_rgb = rgb[non_dark].astype(np.float64) / 255.0
        np.add.at(color_sum, keep_idx, keep_rgb)
        np.add.at(color_cnt, keep_idx, 1)

    has_any = color_cnt > 0
    if not has_any.any():
        return False

    out_colors = np.empty((n, 3), dtype=np.float64)
    out_colors[has_any] = color_sum[has_any] / color_cnt[has_any, None]

    # Fill non-sampled vertices via nearest neighbour from sampled ones so the GLB
    # never has black holes where projection happened to miss.
    if (~has_any).any() and has_any.any():
        sampled_pcd = o3d.geometry.PointCloud()
        sampled_pcd.points = o3d.utility.Vector3dVector(verts[has_any])
        sampled_pcd.colors = o3d.utility.Vector3dVector(out_colors[has_any])
        tree = o3d.geometry.KDTreeFlann(sampled_pcd)
        sampled_cols = np.asarray(sampled_pcd.colors)
        for i in np.where(~has_any)[0]:
            _k, idx, _d = tree.search_knn_vector_3d(verts[i], 1)
            if len(idx):
                out_colors[i] = sampled_cols[idx[0]]
            else:
                out_colors[i] = (0.85, 0.85, 0.85)

    mesh.vertex_colors = o3d.utility.Vector3dVector(out_colors)
    return True
