from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


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

    tree = o3d.geometry.KDTreeFlann(pcd)
    nn_colors = np.empty_like(verts)
    for i in range(verts.shape[0]):
        _, idx, _ = tree.search_knn_vector_3d(verts[i], 1)
        nn_colors[i] = cols[idx[0]] if idx else (0.85, 0.85, 0.85)
    mesh.vertex_colors = o3d.utility.Vector3dVector(nn_colors)
    return mesh


def export_glb(mesh: o3d.geometry.TriangleMesh, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    if vertices.size == 0 or faces.size == 0:
        raise RuntimeError("Cannot export empty mesh")

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

