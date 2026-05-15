"""CPU-friendly export of a 3D Gaussian Splatting ``.ply`` from sparse colored points.

This is **not** the result of optimizing Gaussians with the official CUDA trainer.
It builds one isotropic Gaussian per COLMAP-text sparse point (MapAnything seed) with RGB encoded in
the DC spherical-harmonics bands — enough for many `.ply` viewers to show **color**
without a GPU.

Used when ``torch.cuda.is_available()`` is false and ``gs_allow_cpu_fallback`` is true.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d


# Official 3DGS SH conversion (RGB in [0,1] linear).
_C0 = 0.28209479177387814


def rgb_to_sh_dc(rgb_uint8: np.ndarray) -> np.ndarray:
    """(N, 3) uint8 → (N, 3) float32 SH DC coefficients."""
    rgb = rgb_uint8.astype(np.float32) / 255.0
    return (rgb - 0.5) / _C0


def write_gaussian_ply_from_colored_points(
    xyz: np.ndarray,
    rgb_uint8: np.ndarray,
    output_path: Path,
    *,
    max_points: int = 250_000,
) -> Path:
    """Write binary little-endian 3DGS-style PLY (degree-3, 45 f_rest slots)."""
    if xyz.shape[0] != rgb_uint8.shape[0]:
        raise ValueError("xyz and rgb must have the same row count")
    if xyz.shape[0] == 0:
        raise RuntimeError("No points to export as Gaussian PLY")

    n = xyz.shape[0]
    if n > max_points:
        idx = np.random.choice(n, max_points, replace=False)
        xyz = xyz[idx]
        rgb_uint8 = rgb_uint8[idx]
        n = max_points

    xyz = xyz.astype(np.float32)
    nrm = np.zeros((n, 3), dtype=np.float32)
    f_dc = rgb_to_sh_dc(rgb_uint8).astype(np.float32)
    f_rest = np.zeros((n, 45), dtype=np.float32)

    # Per-point scale from local NN spacing (Open3D KD-tree on float64 points).
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    tree = o3d.geometry.KDTreeFlann(pcd)
    scales = np.empty((n, 3), dtype=np.float32)
    k_nn = min(8, n)
    for i in range(n):
        _, _idxs, dists = tree.search_knn_vector_3d(pcd.points[i], k_nn)
        pos = [float(d) for d in np.asarray(dists).flatten() if float(d) > 1e-18]
        if pos:
            d = float(np.sqrt(min(pos)))
        else:
            d = 0.02
        s_log = float(np.log(max(d * 0.25, 1e-6)))
        scales[i, :] = s_log

    opacity = np.full((n, 1), 0.0, dtype=np.float32)  # sigmoid(0)=0.5
    rot = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1))

    rows = np.concatenate([xyz, nrm, f_dc, f_rest, opacity, scales, rot], axis=1).astype(np.float32)

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property float f_dc_0",
        "property float f_dc_1",
        "property float f_dc_2",
        *[f"property float f_rest_{i}" for i in range(45)],
        "property float opacity",
        "property float scale_0",
        "property float scale_1",
        "property float scale_2",
        "property float rot_0",
        "property float rot_1",
        "property float rot_2",
        "property float rot_3",
        "end_header",
        "",
    ]
    header = "\n".join(header_lines).encode("ascii")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(header)
        f.write(rows.tobytes(order="C"))
    return output_path
