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


def export_glb(mesh: o3d.geometry.TriangleMesh, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    if vertices.size == 0 or faces.size == 0:
        raise RuntimeError("Cannot export empty mesh")

    tri = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    tri.export(str(output_path))
    return output_path

