from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

_log = logging.getLogger(__name__)


def poisson_mesh(
    pcd: o3d.geometry.PointCloud,
    *,
    depth: int = 9,
    density_quantile: float = 0.02,
) -> o3d.geometry.TriangleMesh:
    pcd.estimate_normals()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    if len(mesh.vertices) == 0:
        raise RuntimeError("Poisson meshing produced an empty mesh")

    dens = np.asarray(densities)
    threshold = np.quantile(dens, density_quantile)
    keep_vertices = dens >= threshold
    mesh.remove_vertices_by_mask(~keep_vertices)
    mesh.compute_vertex_normals()
    return mesh


def decimate(mesh: o3d.geometry.TriangleMesh, target_triangles: int) -> o3d.geometry.TriangleMesh:
    tri_count = len(mesh.triangles)
    if tri_count <= target_triangles:
        return mesh
    simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
    simplified.compute_vertex_normals()
    return simplified


def keep_largest_mesh_component(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Drop disconnected fragments; keep the largest triangle component."""
    if len(mesh.triangles) == 0:
        return mesh
    labels, tri_counts, _areas = mesh.cluster_connected_triangles()
    labels = np.asarray(labels)
    tri_counts = np.asarray(tri_counts)
    if labels.size == 0 or tri_counts.size == 0:
        return mesh
    keep_label = int(np.argmax(tri_counts))
    remove_triangles = labels != keep_label
    if not remove_triangles.any():
        return mesh
    out = o3d.geometry.TriangleMesh(mesh)
    out.remove_triangles_by_mask(remove_triangles)
    out.remove_unreferenced_vertices()
    out.compute_vertex_normals()
    return out


def center_and_scale_mesh(mesh: o3d.geometry.TriangleMesh, *, target_extent: float = 1.8) -> o3d.geometry.TriangleMesh:
    """Recentre mesh to origin and scale to a stable viewer-friendly extent."""
    verts = np.asarray(mesh.vertices)
    if verts.size == 0:
        return mesh
    mn = verts.min(axis=0)
    mx = verts.max(axis=0)
    center = (mn + mx) * 0.5
    extent = float(np.max(mx - mn))
    if extent <= 1e-9:
        return mesh
    scale = float(target_extent) / extent
    centered = (verts - center) * scale
    out = o3d.geometry.TriangleMesh(mesh)
    out.vertices = o3d.utility.Vector3dVector(centered)
    out.compute_vertex_normals()
    return out


def autobalance_vertex_colors(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Lift dark/flat vertex colors when reconstruction produced near-black results."""
    has_colors = bool(getattr(mesh, "has_vertex_colors", lambda: False)())
    if not has_colors:
        return mesh
    cols = np.asarray(mesh.vertex_colors, dtype=np.float64)
    if cols.size == 0:
        return mesh

    luma = cols @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
    med = float(np.median(luma))
    if med >= 0.18:
        return mesh

    lo = np.percentile(cols, 1, axis=0)
    hi = np.percentile(cols, 99, axis=0)
    denom = np.maximum(hi - lo, 1e-6)
    balanced = (cols - lo) / denom
    balanced = np.clip(balanced, 0.0, 1.0)
    balanced = np.power(balanced, 0.8)

    out = o3d.geometry.TriangleMesh(mesh)
    out.vertex_colors = o3d.utility.Vector3dVector(balanced)
    return out


def transfer_vertex_colors_from_point_cloud(
    mesh: o3d.geometry.TriangleMesh,
    pcd: o3d.geometry.PointCloud,
) -> o3d.geometry.TriangleMesh:
    """Assign each mesh vertex the color of its nearest point in ``pcd``.

    Used as a deterministic fallback because Open3D's Poisson and
    ``simplify_quadric_decimation`` do not always preserve vertex colors across
    versions/builds — we do it explicitly so the GLB always has photo color.
    """
    if not pcd.has_colors():
        return mesh
    verts = np.asarray(mesh.vertices)
    if verts.size == 0:
        return mesh
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors)
    if pts.size == 0 or cols.size == 0:
        return mesh

    pts64 = np.asarray(pts, dtype=np.float64)
    verts64 = np.asarray(verts, dtype=np.float64)
    cols64 = np.asarray(cols, dtype=np.float64)

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(pts64)
        try:
            _, idx = tree.query(verts64, k=1, workers=-1)
        except TypeError:
            _, idx = tree.query(verts64, k=1)
        idx = np.asarray(idx, dtype=np.intp).reshape(-1)
        nn_colors = cols64[idx]
    except ImportError:
        _log.warning(
            "scipy is not installed; mesh vertex colour transfer falls back to a slow per-vertex loop "
            "(install scipy for large meshes)."
        )
        tree = o3d.geometry.KDTreeFlann(pcd)
        nn_colors = np.empty_like(verts64)
        for i in range(verts64.shape[0]):
            _, idx, _ = tree.search_knn_vector_3d(verts64[i], 1)
            nn_colors[i] = cols64[idx[0]] if idx else (0.85, 0.85, 0.85)

    mesh.vertex_colors = o3d.utility.Vector3dVector(nn_colors)
    return mesh


def export_glb(
    mesh: o3d.geometry.TriangleMesh,
    output_path: Path,
    *,
    compressed: bool = False,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    if vertices.size == 0 or faces.size == 0:
        raise RuntimeError("Cannot export empty mesh")

    if compressed and output_path.suffix.lower() == ".glb":
        try:
            ok = o3d.io.write_triangle_mesh(
                str(output_path),
                mesh,
                write_ascii=False,
                compressed=True,
                write_vertex_normals=True,
                write_vertex_colors=True,
                write_triangle_uvs=False,
            )
            if ok and output_path.is_file() and output_path.stat().st_size > 256:
                return output_path
        except Exception as exc:
            _log.warning("Open3D compressed GLB failed (%s); using trimesh export.", exc)

    tri = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # Carry per-vertex colors from Open3D (Poisson interpolates input pcd colors,
    # decimation preserves them). Fallback to neutral gray when missing so viewers
    # that default to a dark PBR material don't render the mesh as black.
    has_colors = bool(getattr(mesh, "has_vertex_colors", lambda: False)())
    vc = np.asarray(mesh.vertex_colors) if has_colors else None
    if vc is not None and vc.size > 0 and vc.shape[0] == vertices.shape[0]:
        rgba = np.empty((vc.shape[0], 4), dtype=np.uint8)
        rgba[:, :3] = np.clip(vc * 255.0, 0, 255).astype(np.uint8)
        rgba[:, 3] = 255
        tri.visual.vertex_colors = rgba
    else:
        n = len(vertices)
        tri.visual.vertex_colors = np.full((n, 4), [220, 220, 220, 255], dtype=np.uint8)

    tri.export(str(output_path))
    return output_path

