from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from .runtime_settings import RuntimeSettings


@dataclass
class ShapePriorReport:
    applied: bool
    label: str | None
    confidence: float
    template_path: str | None
    alignment_rmse: float | None
    reason: str | None = None


def _resolve_template_root(settings: RuntimeSettings) -> Path | None:
    raw = (getattr(settings, "shape_prior_template_root", None) or "").strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p


def _list_templates(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not root.is_dir():
        return out
    for ext in ("*.glb", "*.gltf", "*.ply", "*.obj", "*.stl"):
        for p in root.glob(ext):
            label = p.stem.strip().lower()
            if label and label not in out:
                out[label] = p
    return out


def _classify_label(image_paths: list[Path], labels: list[str], settings: RuntimeSettings) -> tuple[str | None, float]:
    forced = (getattr(settings, "shape_prior_force_label", None) or "").strip().lower()
    if forced:
        if forced in labels:
            return forced, 1.0
        return None, 0.0

    # Optional CLIP path (best effort): if unavailable, no-op.
    try:
        import torch
        import open_clip
        from PIL import Image
    except Exception:
        return None, 0.0

    if not image_paths or not labels:
        return None, 0.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    except Exception:
        return None, 0.0
    model = model.to(device)
    model.eval()

    sample_paths = image_paths[: min(len(image_paths), 8)]
    imgs = []
    for p in sample_paths:
        try:
            imgs.append(preprocess(Image.open(p).convert("RGB")))
        except Exception:
            continue
    if not imgs:
        return None, 0.0

    with torch.no_grad():
        image_tensor = torch.stack(imgs).to(device)
        texts = [f"a centered product photo of a {lb}" for lb in labels]
        text_tokens = open_clip.tokenize(texts).to(device)
        im_feat = model.encode_image(image_tensor)
        tx_feat = model.encode_text(text_tokens)
        im_feat = im_feat / im_feat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        tx_feat = tx_feat / tx_feat.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        sims = im_feat @ tx_feat.T
        mean_sims = sims.mean(dim=0)
        probs = torch.softmax(mean_sims * 8.0, dim=0)
        idx = int(torch.argmax(probs).item())
        conf = float(probs[idx].item())
    return labels[idx], conf


def _mesh_norm(mesh: o3d.geometry.TriangleMesh) -> tuple[np.ndarray, np.ndarray, float]:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    mn = verts.min(axis=0)
    mx = verts.max(axis=0)
    center = (mn + mx) * 0.5
    extent = float(np.max(mx - mn))
    extent = max(extent, 1e-6)
    norm = (verts - center) / extent
    return norm, center, extent


def _nearest_points(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(dst.astype(np.float64))
    tree = o3d.geometry.KDTreeFlann(pcd)
    out = np.empty_like(src, dtype=np.float64)
    for i in range(src.shape[0]):
        _, idx, _ = tree.search_knn_vector_3d(src[i].astype(np.float64), 1)
        out[i] = dst[idx[0]] if idx else src[i]
    return out


def apply_shape_prior_correction(
    mesh: o3d.geometry.TriangleMesh,
    image_paths: list[Path],
    settings: RuntimeSettings,
) -> tuple[o3d.geometry.TriangleMesh, ShapePriorReport]:
    if not bool(getattr(settings, "shape_prior_enabled", False)):
        return mesh, ShapePriorReport(False, None, 0.0, None, None, reason="shape_prior_enabled=false")

    root = _resolve_template_root(settings)
    if root is None:
        return mesh, ShapePriorReport(False, None, 0.0, None, None, reason="shape_prior_template_root is empty")
    templates = _list_templates(root)
    if not templates:
        return mesh, ShapePriorReport(False, None, 0.0, None, None, reason=f"no templates found in {root}")

    labels = sorted(templates.keys())
    label, conf = _classify_label(image_paths, labels, settings)
    min_conf = float(getattr(settings, "shape_prior_min_confidence", 0.34))
    if not label or conf < min_conf:
        return mesh, ShapePriorReport(False, label, conf, None, None, reason="classification confidence below threshold")

    tpl_path = templates.get(label)
    if tpl_path is None:
        return mesh, ShapePriorReport(False, label, conf, None, None, reason="selected template not found")

    tpl = o3d.io.read_triangle_mesh(str(tpl_path))
    if tpl is None or len(tpl.vertices) < 16:
        return mesh, ShapePriorReport(False, label, conf, str(tpl_path), None, reason="template mesh unreadable/empty")

    src_norm, src_center, src_extent = _mesh_norm(mesh)
    tpl_norm, _tpl_center, _tpl_extent = _mesh_norm(tpl)

    src_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src_norm))
    tpl_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(tpl_norm))
    try:
        reg = o3d.pipelines.registration.registration_icp(
            tpl_pcd,
            src_pcd,
            0.15,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
        rmse = float(reg.inlier_rmse)
        T = reg.transformation
    except Exception:
        return mesh, ShapePriorReport(False, label, conf, str(tpl_path), None, reason="ICP alignment failed")

    max_rmse = float(getattr(settings, "shape_prior_max_alignment_rmse", 0.12))
    if rmse > max_rmse:
        return mesh, ShapePriorReport(False, label, conf, str(tpl_path), rmse, reason="alignment RMSE too high")

    tpl_pts = np.asarray(tpl_pcd.points, dtype=np.float64)
    if tpl_pts.size == 0:
        return mesh, ShapePriorReport(False, label, conf, str(tpl_path), rmse, reason="template points empty")

    tpl_pts_h = np.concatenate([tpl_pts, np.ones((tpl_pts.shape[0], 1), dtype=np.float64)], axis=1)
    tpl_aligned = (T @ tpl_pts_h.T).T[:, :3]

    nn = _nearest_points(src_norm, tpl_aligned)
    deform = float(getattr(settings, "shape_prior_deform_strength", 0.22))
    keep_detail = float(getattr(settings, "shape_prior_preserve_detail", 0.78))
    alpha = float(np.clip(deform * (1.0 - 0.5 * keep_detail), 0.0, 1.0))
    corrected_norm = (1.0 - alpha) * src_norm + alpha * nn
    corrected_verts = corrected_norm * src_extent + src_center

    out = o3d.geometry.TriangleMesh(mesh)
    out.vertices = o3d.utility.Vector3dVector(corrected_verts.astype(np.float64))
    out.compute_vertex_normals()
    return out, ShapePriorReport(True, label, conf, str(tpl_path), rmse, reason=None)

