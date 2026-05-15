"""MapAnything feed-forward reconstruction (Facebook Research) → point cloud + cameras."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .multiview_scene import MultiviewMetricScene
from .runtime_settings import RuntimeSettings


def _resolve_torch_device(settings: RuntimeSettings):
    import torch

    device_str = settings.device
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _align_hw_mask(mask_np: np.ndarray, h: int, w: int) -> np.ndarray:
    """Squeeze/normalize mask to 2D and resize with nearest-neighbor to match pts (H,W)."""
    import cv2

    m = np.asarray(mask_np, dtype=np.float32)
    m = np.squeeze(m)
    while m.ndim > 2 and m.shape[-1] == 1:
        m = m[..., 0]
    if m.ndim != 2:
        if m.size == h * w:
            m = m.reshape(h, w)
        else:
            return np.ones((h, w), dtype=bool)
    if m.shape[0] != h or m.shape[1] != w:
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return m > 0.5


def _align_hw_rgb(rgb: np.ndarray, h: int, w: int) -> np.ndarray:
    """Resize RGB to (H,W,3); MapAnything outputs can differ from ``pts3d`` grid."""
    import cv2

    x = np.asarray(rgb, dtype=np.float32)
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3 and x.shape[-1] >= 3:
        rgb3 = x[..., :3]
    elif x.ndim == 2:
        return np.full((h, w, 3), 200, dtype=np.uint8)
    else:
        return np.full((h, w, 3), 200, dtype=np.uint8)

    if rgb3.shape[0] == h and rgb3.shape[1] == w:
        return np.clip(np.round(rgb3), 0, 255).astype(np.uint8)

    bgr = cv2.cvtColor(rgb3, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    out = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def _subsample_paths(paths: list[Path], max_n: int) -> list[Path]:
    if len(paths) <= max_n:
        return list(paths)
    idx = np.linspace(0, len(paths) - 1, max_n)
    idx = np.unique(np.round(idx).astype(int))
    out = [paths[int(i)] for i in idx]
    if len(out) < 2:
        return [paths[0], paths[-1]]
    return out


def run_mapanything_scene(masked_image_paths: list[Path], settings: RuntimeSettings) -> MultiviewMetricScene:
    if len(masked_image_paths) < 2:
        raise ValueError("MapAnything needs at least 2 images")

    try:
        import torch
        from mapanything.models import MapAnything
        from mapanything.utils.image import load_images
    except ImportError as exc:
        raise RuntimeError(
            "MapAnything is not installed. Install the fork from Meta, e.g.:\n"
            "  pip install \"git+https://github.com/facebookresearch/map-anything.git\"\n"
            "CUDA PyTorch recommended. Original error: " + str(exc)
        ) from exc

    paths = sorted(masked_image_paths)
    max_v = int(getattr(settings, "mapanything_max_input_views", 48))
    working_paths = _subsample_paths(paths, max_v)
    paths_str = [str(p.resolve()) for p in working_paths]

    device = _resolve_torch_device(settings)
    pretrained_id = (getattr(settings, "mapanything_pretrained_id", None) or "facebook/map-anything-apache").strip()
    model = MapAnything.from_pretrained(pretrained_id).to(device)
    model.eval()

    views = load_images(paths_str)

    mem_eff = bool(getattr(settings, "mapanything_memory_efficient_inference", True))
    mini = int(settings.mapanything_minibatch_size)
    use_amp = bool(getattr(settings, "mapanything_use_amp", True)) and device.type == "cuda"
    _bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
    bf16_ok = bool(_bf16_supported()) if callable(_bf16_supported) else False
    if use_amp and device.type == "cuda" and bf16_ok:
        amp_dtype = "bf16"
    elif use_amp and device.type == "cuda":
        amp_dtype = "fp16"
    else:
        use_amp = False
        amp_dtype = "fp32"

    with torch.no_grad():
        preds = model.infer(
            views,
            memory_efficient_inference=mem_eff,
            minibatch_size=mini if mem_eff else None,
            use_amp=use_amp,
            amp_dtype=amp_dtype if use_amp else "fp32",
            apply_mask=bool(getattr(settings, "mapanything_apply_mask", True)),
            mask_edges=bool(getattr(settings, "mapanything_mask_edges", True)),
            apply_confidence_mask=bool(getattr(settings, "mapanything_apply_confidence_mask", False)),
            confidence_percentile=int(getattr(settings, "mapanything_confidence_percentile", 10)),
            use_multiview_confidence=bool(getattr(settings, "mapanything_use_multiview_confidence", False)),
        )

    xyz_blocks: list[np.ndarray] = []
    rgb_blocks: list[np.ndarray] = []
    image_sizes: list[tuple[int, int]] = []
    intrinsics: list[np.ndarray] = []
    poses_w2c: list[np.ndarray] = []
    poses_c2w: list[np.ndarray] = []

    for vi, pred in enumerate(preds):
        img_path = working_paths[vi]
        pts_t = pred["pts3d"]
        if hasattr(pts_t, "detach"):
            pts = pts_t.detach().float().cpu().numpy()
        else:
            pts = np.asarray(pts_t, dtype=np.float32)
        if pts.ndim == 4:
            pts = pts[0]
        h, w, _ = pts.shape

        mask_t = pred.get("mask")
        if mask_t is not None:
            if hasattr(mask_t, "detach"):
                m = mask_t.detach().float().cpu().numpy()
            else:
                m = np.asarray(mask_t, dtype=np.float32)
            keep = _align_hw_mask(m, h, w)
        else:
            keep = np.ones((h, w), dtype=bool)

        rgb_t = pred.get("img_no_norm")
        if rgb_t is None:
            rgb = np.full((h, w, 3), 200, dtype=np.uint8)
        else:
            if hasattr(rgb_t, "detach"):
                rgb_raw = rgb_t.detach().float().cpu().numpy()
            else:
                rgb_raw = np.asarray(rgb_t, dtype=np.float32)
            rgb = _align_hw_rgb(rgb_raw, h, w)

        flat_keep = keep.reshape(-1)
        flat_pts = pts.reshape(-1, 3).astype(np.float32, copy=False)
        flat_rgb = rgb.reshape(-1, 3)
        xyz_blocks.append(flat_pts[flat_keep])
        rgb_blocks.append(flat_rgb[flat_keep])

        k_t = pred["intrinsics"]
        if hasattr(k_t, "detach"):
            K = k_t.detach().float().cpu().numpy()
        else:
            K = np.asarray(k_t, dtype=np.float64)
        if K.ndim == 3:
            K = K[0]
        intrinsics.append(K.astype(np.float64, copy=False))
        image_sizes.append((int(w), int(h)))

        c2w_t = pred["camera_poses"]
        if hasattr(c2w_t, "detach"):
            c2w = c2w_t.detach().float().cpu().numpy()
        else:
            c2w = np.asarray(c2w_t, dtype=np.float64)
        if c2w.ndim == 3:
            c2w = c2w[0]
        poses_c2w.append(c2w.astype(np.float64, copy=False))
        w2c = np.linalg.inv(c2w)
        poses_w2c.append(w2c.astype(np.float64, copy=False))

    if not xyz_blocks:
        raise RuntimeError("MapAnything produced no valid 3D points (empty mask).")

    points = np.concatenate(xyz_blocks, axis=0)
    colors = np.concatenate(rgb_blocks, axis=0)

    return MultiviewMetricScene(
        points=points,
        colors=colors,
        image_paths=working_paths,
        image_sizes=image_sizes,
        intrinsics=intrinsics,
        poses_w2c=poses_w2c,
        poses_c2w=poses_c2w,
    )
