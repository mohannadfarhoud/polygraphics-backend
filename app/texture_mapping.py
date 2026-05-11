"""Multi-view diffuse texture baking for mesh GLB export (CPU-friendly).

When Gaussian Splatting is unavailable (no GPU), this improves realism vs plain
vertex colours by:

1. UV-unwrapping the triangle mesh with **xatlas** (optional dependency).
2. Rasterizing each UV triangle into an atlas and, per texel, recovering the
   corresponding 3D point + interpolated normal.
3. Projecting that point into each calibrated camera, sampling the **original**
   (unmasked) photo, and blending samples with a Lambert-like weight
   ``max(0, n·v)`` where ``v`` is the direction toward the camera.

The result is a ``trimesh.Trimesh`` with ``TextureVisuals`` suitable for GLB.
If ``xatlas`` is missing or unwrap fails, returns ``None`` and the pipeline
falls back to vertex-colour ``export_glb``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import trimesh

from .color_baking import CameraView


def _read_rgb(path: Path, size: tuple[int, int]) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    W, H = int(size[0]), int(size[1])
    if (img.shape[1], img.shape[0]) != (W, H):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _barycentric_2d(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> tuple[float, float, float] | None:
    """Return (w,u,v) with P = w*A + u*B + v*C if P is inside triangle ABC in 2D, else None."""
    denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(denom) < 1e-12:
        return None
    wa = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / denom
    wb = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / denom
    wc = 1.0 - wa - wb
    if wa >= -1e-6 and wb >= -1e-6 and wc >= -1e-6:
        return float(wa), float(wb), float(wc)
    return None


def _sample_multiview(
    P: np.ndarray,
    N: np.ndarray,
    views: list[CameraView],
    images: list[np.ndarray | None],
    *,
    skip_dark_threshold: int,
) -> np.ndarray | None:
    """Weighted average RGB in [0,1]^3, or None if no valid sample."""
    homog = np.array([P[0], P[1], P[2], 1.0], dtype=np.float64)
    acc = np.zeros(3, dtype=np.float64)
    wsum = 0.0
    N = N / (np.linalg.norm(N) + 1e-9)

    for v, img in zip(views, images):
        if img is None:
            continue
        W, H = int(v.image_size[0]), int(v.image_size[1])
        w2c = np.asarray(v.w2c, dtype=np.float64)
        K = np.asarray(v.K, dtype=np.float64)
        pcam = w2c @ homog
        z = float(pcam[2])
        if z <= 1e-6:
            continue
        x = float(pcam[0]) / z
        y = float(pcam[1]) / z
        u_pix = K[0, 0] * x + K[0, 2]
        v_pix = K[1, 1] * y + K[1, 2]
        ui = int(round(u_pix))
        vi = int(round(v_pix))
        if ui < 0 or ui >= W or vi < 0 or vi >= H:
            continue
        rgb = img[vi, ui]
        if int(rgb.sum()) <= skip_dark_threshold:
            continue
        try:
            c2w = np.linalg.inv(w2c)
        except np.linalg.LinAlgError:
            continue
        cam_c = c2w[:3, 3]
        view_vec = cam_c - P
        dist = np.linalg.norm(view_vec)
        if dist < 1e-9:
            continue
        view_vec = view_vec / dist
        ndot = max(0.0, float(np.dot(N, view_vec)))
        if ndot < 1e-8:
            continue
        wgt = ndot
        acc += rgb.astype(np.float64) / 255.0 * wgt
        wsum += wgt

    if wsum <= 1e-12:
        return None
    return acc / wsum


def build_textured_trimesh(
    mesh: o3d.geometry.TriangleMesh,
    views: list[CameraView],
    *,
    atlas_size: int = 2048,
    skip_dark_threshold: int = 8,
    flip_uv_v: bool = True,
) -> trimesh.Trimesh | None:
    """Return a textured ``trimesh.Trimesh``, or ``None`` if unwrap/baking cannot run."""
    if not views:
        return None

    try:
        import xatlas  # type: ignore[import-untyped]
    except ImportError:
        return None

    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.uint32)
    if verts.size == 0 or faces.size == 0:
        return None

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    norms = np.asarray(mesh.vertex_normals, dtype=np.float64)

    try:
        vmapping, indices_out, uvs = xatlas.parametrize(verts, faces)
    except Exception:
        return None

    vmapping = np.asarray(vmapping, dtype=np.int64)
    indices_out = np.asarray(indices_out, dtype=np.int64)
    uvs = np.asarray(uvs, dtype=np.float64)
    if uvs.ndim != 2 or uvs.shape[1] != 2:
        return None

    new_verts = verts[vmapping]
    new_norms = norms[vmapping]

    if indices_out.ndim == 1:
        tris = indices_out.reshape(-1, 3)
    else:
        tris = indices_out

    images: list[np.ndarray | None] = [_read_rgb(v.image_path, v.image_size) for v in views]

    H = W = int(atlas_size)
    rgb_sum = np.zeros((H, W, 3), dtype=np.float64)
    w_sum = np.zeros((H, W), dtype=np.float64)

    scale = float(atlas_size - 1)

    for ti in range(tris.shape[0]):
        ia, ib, ic = int(tris[ti, 0]), int(tris[ti, 1]), int(tris[ti, 2])
        uv_a = uvs[ia].copy()
        uv_b = uvs[ib].copy()
        uv_c = uvs[ic].copy()
        if flip_uv_v:
            uv_a[1] = 1.0 - uv_a[1]
            uv_b[1] = 1.0 - uv_b[1]
            uv_c[1] = 1.0 - uv_c[1]

        pa = uv_a * scale
        pb = uv_b * scale
        pc = uv_c * scale

        xmin = int(np.floor(min(pa[0], pb[0], pc[0])))
        xmax = int(np.ceil(max(pa[0], pb[0], pc[0])))
        ymin = int(np.floor(min(pa[1], pb[1], pc[1])))
        ymax = int(np.ceil(max(pa[1], pb[1], pc[1])))
        xmin = max(0, min(W - 1, xmin))
        xmax = max(0, min(W - 1, xmax))
        ymin = max(0, min(H - 1, ymin))
        ymax = max(0, min(H - 1, ymax))

        va = new_verts[ia].astype(np.float64)
        vb = new_verts[ib].astype(np.float64)
        vc = new_verts[ic].astype(np.float64)
        na = new_norms[ia]
        nb = new_norms[ib]
        nc = new_norms[ic]

        for yy in range(ymin, ymax + 1):
            for xx in range(xmin, xmax + 1):
                p2 = np.array([xx + 0.5, yy + 0.5], dtype=np.float64)
                bar = _barycentric_2d(p2, pa, pb, pc)
                if bar is None:
                    continue
                wa, wb, wc = bar
                P = wa * va + wb * vb + wc * vc
                N = wa * na + wb * nb + wc * nc
                col = _sample_multiview(P, N, views, images, skip_dark_threshold=skip_dark_threshold)
                if col is None:
                    continue
                rgb_sum[yy, xx] += col
                w_sum[yy, xx] += 1.0

    valid = w_sum > 1e-9
    if not np.any(valid):
        return None

    tex = np.zeros((H, W, 3), dtype=np.float32)
    tex[valid] = (rgb_sum[valid] / w_sum[valid, np.newaxis]).astype(np.float32)
    tex_u8 = np.clip(tex * 255.0, 0, 255).astype(np.uint8)

    holes = (~valid).astype(np.uint8) * 255
    if np.any(holes):
        bgr = cv2.cvtColor(tex_u8, cv2.COLOR_RGB2BGR)
        inp = cv2.inpaint(bgr, holes, 5, cv2.INPAINT_TELEA)
        tex_u8 = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)

    # Trimesh texture UVs should match new_verts row count
    uv_out = uvs.copy()
    if flip_uv_v:
        uv_out[:, 1] = 1.0 - uv_out[:, 1]

    tri_tm = trimesh.Trimesh(
        vertices=new_verts,
        faces=tris,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uv_out, image=tex_u8),
    )
    return tri_tm


def export_textured_glb(tm: trimesh.Trimesh, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tm.export(str(output_path))
    return output_path
