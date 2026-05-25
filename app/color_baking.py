"""Bake mesh-vertex colors from original photos using camera projections.

Given a list of camera views ``(image_path, image_size, K, w2c)`` aligned to the
mesh's world frame, project each mesh vertex into every image, sample the colour,
and average across visible views. This produces photo-true vertex colours even
when the upstream point-cloud colours were stripped, normalised, or otherwise
came out near-black.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

_log = logging.getLogger(__name__)


@dataclass
class CameraView:
    image_path: Path  # original (preferably unmasked) image on disk
    image_size: tuple[int, int]  # (W, H) the K below was estimated for
    K: np.ndarray  # 3x3 intrinsics matrix (float64)
    w2c: np.ndarray  # 4x4 world->camera transform (float64)
    surface_region_confidence: float = 1.0
    surface_region_label: str | None = None


@dataclass
class BakeDiagnostics:
    per_view_region_confidence: dict[str, float]
    region_projection_coverage: dict[str, float | int]
    dominant_surface_regions_detected: int


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
    blend_weight: float = 0.65,
    min_confidence: float = 0.60,
    seam_smoothing: float = 0.55,
) -> tuple[bool, BakeDiagnostics]:
    """Sample colours per-vertex from ``views`` and overwrite ``mesh.vertex_colors``.

    Pixels darker than ``skip_dark_threshold`` (sum of channels) are skipped so
    masked-out (black) backgrounds don't pollute the average. Returns ``True``
    when at least one vertex received a sample, ``False`` otherwise.
    """
    verts = np.asarray(mesh.vertices)
    if verts.size == 0 or not views:
        return False, BakeDiagnostics({}, {"mesh_area_ratio": 0.0, "regions_projected": 0}, 0)

    images: list[np.ndarray | None] = [_read_image_at(v.image_path, v.image_size) for v in views]

    n = verts.shape[0]
    color_sum = np.zeros((n, 3), dtype=np.float64)
    weight_sum = np.zeros(n, dtype=np.float64)
    homog = np.concatenate([verts, np.ones((n, 1), dtype=np.float64)], axis=1)  # Nx4
    projected_vertices = np.zeros(n, dtype=np.bool_)
    per_view_conf: dict[str, float] = {}
    regions_projected = 0

    for view, img in zip(views, images):
        if img is None:
            continue
        view_conf = float(np.clip(getattr(view, "surface_region_confidence", 1.0), 0.0, 1.0))
        if view_conf < min_confidence:
            continue
        per_view_conf[Path(view.image_path).name] = view_conf
        regions_projected += 1
        W, H = int(view.image_size[0]), int(view.image_size[1])

        pcam = (np.asarray(view.w2c, dtype=np.float64) @ homog.T).T  # Nx4
        z = pcam[:, 2]
        valid_z = z > 1e-6
        if not valid_z.any():
            continue

        K = np.asarray(view.K, dtype=np.float64)
        x = pcam[valid_z, 0] / z[valid_z]
        y = pcam[valid_z, 1] / z[valid_z]
        u = K[0, 0] * x + K[0, 2]
        vc = K[1, 1] * y + K[1, 2]

        idx_valid = np.where(valid_z)[0]
        in_bounds = (u >= 0) & (u < W) & (vc >= 0) & (vc < H)
        if not in_bounds.any():
            continue

        global_idx = idx_valid[in_bounds]
        u_b = u[in_bounds].astype(np.float64)
        vc_b = vc[in_bounds].astype(np.float64)

        x0 = np.floor(u_b).astype(np.int32)
        y0 = np.floor(vc_b).astype(np.int32)
        x1 = np.minimum(x0 + 1, W - 1)
        y1 = np.minimum(y0 + 1, H - 1)
        wx = u_b - x0.astype(np.float64)
        wy = vc_b - y0.astype(np.float64)

        I00 = img[y0, x0].astype(np.float64)
        I01 = img[y0, x1].astype(np.float64)
        I10 = img[y1, x0].astype(np.float64)
        I11 = img[y1, x1].astype(np.float64)
        rgb = (
            (1.0 - wx)[:, None] * (1.0 - wy)[:, None] * I00
            + wx[:, None] * (1.0 - wy)[:, None] * I01
            + (1.0 - wx)[:, None] * wy[:, None] * I10
            + wx[:, None] * wy[:, None] * I11
        )
        sums = rgb.sum(axis=1)
        non_dark = sums > int(skip_dark_threshold)
        if not non_dark.any():
            continue

        keep_idx = global_idx[non_dark]
        keep_rgb = rgb[non_dark] / 255.0
        base_w = float(np.clip(blend_weight, 0.0, 1.0))
        conf_w = (1.0 - base_w) + (base_w * view_conf)
        z_local = z[keep_idx]
        z_norm = z_local / max(1e-6, float(np.percentile(z_local, 95)))
        z_w = np.clip(1.0 - z_norm * float(np.clip(seam_smoothing, 0.0, 1.0)), 0.2, 1.0)
        sample_w = conf_w * z_w
        np.add.at(color_sum, keep_idx, keep_rgb * sample_w[:, None])
        np.add.at(weight_sum, keep_idx, sample_w)
        projected_vertices[keep_idx] = True

    has_any = weight_sum > 1e-8
    if not has_any.any():
        return False, BakeDiagnostics(
            per_view_conf,
            {"mesh_area_ratio": 0.0, "regions_projected": int(regions_projected)},
            int(regions_projected),
        )

    out_colors = np.empty((n, 3), dtype=np.float64)
    out_colors[has_any] = color_sum[has_any] / weight_sum[has_any, None]

    # Fill non-sampled vertices via nearest neighbour from sampled ones so the GLB
    # never has black holes where projection happened to miss.
    if (~has_any).any() and has_any.any():
        src_pts = verts[has_any].astype(np.float64)
        src_cols = out_colors[has_any]
        miss = np.where(~has_any)[0]
        dst = verts[miss].astype(np.float64)
        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(src_pts)
            try:
                _, nn_i = tree.query(dst, k=1, workers=-1)
            except TypeError:
                _, nn_i = tree.query(dst, k=1)
            nn_i = np.asarray(nn_i, dtype=np.intp).reshape(-1)
            out_colors[miss] = src_cols[nn_i]
        except ImportError:
            _log.warning(
                "scipy is not installed; photo vertex hole-fill uses a slow loop (install scipy)."
            )
            sampled_pcd = o3d.geometry.PointCloud()
            sampled_pcd.points = o3d.utility.Vector3dVector(src_pts)
            sampled_pcd.colors = o3d.utility.Vector3dVector(src_cols)
            tree = o3d.geometry.KDTreeFlann(sampled_pcd)
            sampled_cols = np.asarray(sampled_pcd.colors)
            for i in miss:
                _k, idx, _d = tree.search_knn_vector_3d(verts[i], 1)
                if len(idx):
                    out_colors[i] = sampled_cols[idx[0]]
                else:
                    out_colors[i] = (0.85, 0.85, 0.85)

    mesh.vertex_colors = o3d.utility.Vector3dVector(out_colors)
    coverage = float(np.count_nonzero(projected_vertices)) / float(max(1, n))
    diagnostics = BakeDiagnostics(
        per_view_region_confidence=per_view_conf,
        region_projection_coverage={
            "mesh_area_ratio": float(np.clip(coverage, 0.0, 1.0)),
            "regions_projected": int(regions_projected),
        },
        dominant_surface_regions_detected=int(regions_projected),
    )
    return True, diagnostics
