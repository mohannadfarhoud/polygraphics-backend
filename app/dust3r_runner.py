from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .runtime_settings import RuntimeSettings


@dataclass
class Dust3rScene:
    """Aggregated DUSt3R global-aligner output, used for both meshing and the COLMAP bridge.

    Per-image arrays are aligned to ``image_paths``. ``points`` is the merged Nx3
    point cloud after the per-pixel confidence filter.
    """

    points: np.ndarray  # (N, 3) float32 — confidence-filtered, world-frame
    colors: np.ndarray  # (N, 3) uint8 — RGB per point (matches ``points``)
    image_paths: list[Path] = field(default_factory=list)
    image_sizes: list[tuple[int, int]] = field(default_factory=list)  # (W, H) per image
    intrinsics: list[np.ndarray] = field(default_factory=list)  # 3x3 K per image
    poses_w2c: list[np.ndarray] = field(default_factory=list)  # 4x4 world->camera
    poses_c2w: list[np.ndarray] = field(default_factory=list)  # 4x4 camera->world


def _resolve_device(settings: RuntimeSettings):
    import torch

    device_str = settings.device
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _load_dust3r():
    try:
        from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
        from dust3r.image_pairs import make_pairs
        from dust3r.inference import inference
        from dust3r.model import AsymmetricCroCo3DStereo
        from dust3r.utils.image import load_images
    except ImportError as exc:
        raise RuntimeError(
            "DUSt3R Python package is not installed. Clone https://github.com/naver/dust3r "
            "and install it in this environment, plus torch/torchvision. "
            "Original error: " + str(exc)
        ) from exc
    return GlobalAlignerMode, global_aligner, make_pairs, inference, AsymmetricCroCo3DStereo, load_images


def _load_model(settings: RuntimeSettings, device):
    from dust3r.model import AsymmetricCroCo3DStereo

    ck_ref = (settings.dust3r_checkpoint_path or "").strip()
    if not ck_ref:
        raise RuntimeError("dust3r_checkpoint_path is empty")
    ck_path = Path(ck_ref)
    load_ref = str(ck_path.resolve()) if ck_path.exists() else ck_ref

    try:
        model = AsymmetricCroCo3DStereo.from_pretrained(load_ref)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load DUSt3R weights from {load_ref!r}. "
            "Use a Hugging Face model id or a local snapshot directory. "
            f"Original error: {exc}"
        ) from exc
    return model.to(device)


def run_dust3r_scene(masked_image_paths: list[Path], settings: RuntimeSettings) -> Dust3rScene:
    """Phase 2 + Phase 3 of the protocol.

    Runs DUSt3R + ``GlobalAligner`` and returns a confidence-filtered point cloud
    together with per-image intrinsics and poses for the COLMAP bridge.
    """
    if len(masked_image_paths) < 2:
        raise ValueError("DUSt3R needs at least 2 images")

    import torch

    (
        GlobalAlignerMode,
        global_aligner,
        make_pairs,
        inference,
        _AsymmetricCroCo3DStereo,
        load_images,
    ) = _load_dust3r()

    paths = [str(p.resolve()) for p in masked_image_paths]
    device = _resolve_device(settings)
    model = _load_model(settings, device)

    max_side = min(settings.max_image_side, 512)
    images = load_images(paths, size=max_side)
    pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)

    batch_size = 1 if device.type == "cpu" else 2
    output = inference(pairs, model, device, batch_size=batch_size)

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.compute_global_alignment(
        init="mst",
        niter=int(settings.dust3r_aligner_iters),
        schedule="cosine",
        lr=float(settings.dust3r_aligner_lr),
    )

    pts_list = scene.get_pts3d()
    masks_list = scene.get_masks() if hasattr(scene, "get_masks") else [None] * len(pts_list)
    conf_list = (
        scene.get_conf() if hasattr(scene, "get_conf") and settings.dust3r_confidence_threshold > 0
        else [None] * len(pts_list)
    )
    rgb_list = scene.imgs if hasattr(scene, "imgs") else [None] * len(pts_list)

    # Cameras
    try:
        focals = scene.get_focals().detach().cpu().numpy().reshape(-1)
    except Exception:
        focals = np.zeros(len(pts_list), dtype=np.float32)
    try:
        c2w_all = scene.get_im_poses().detach().cpu().numpy()  # (N,4,4)
    except Exception:
        c2w_all = np.tile(np.eye(4, dtype=np.float32), (len(pts_list), 1, 1))

    conf_thr = float(settings.dust3r_confidence_threshold)
    chunks_pts: list[np.ndarray] = []
    chunks_rgb: list[np.ndarray] = []

    image_sizes: list[tuple[int, int]] = []
    intrinsics: list[np.ndarray] = []
    poses_w2c: list[np.ndarray] = []
    poses_c2w: list[np.ndarray] = []

    for idx, (pts, mask) in enumerate(zip(pts_list, masks_list)):
        pts_np = pts.detach().cpu().numpy() if hasattr(pts, "detach") else np.asarray(pts)
        if pts_np.ndim == 3 and pts_np.shape[-1] == 3:
            H, W, _ = pts_np.shape
        else:
            # Fallback: assume already flattened (N, 3); use the matching image's RGB shape later.
            n = int(pts_np.size // 3)
            H = int(np.sqrt(max(1, n)))
            W = int(max(1, n // max(1, H)))
        p = pts_np.reshape(-1, 3)
        image_sizes.append((int(W), int(H)))

        rgb = rgb_list[idx] if idx < len(rgb_list) else None
        if rgb is not None:
            rgb_arr = np.asarray(rgb)
            if rgb_arr.dtype != np.uint8:
                rgb_arr = (np.clip(rgb_arr, 0.0, 1.0) * 255.0).astype(np.uint8)
            rgb_arr = rgb_arr.reshape(-1, 3)
        else:
            rgb_arr = np.full((p.shape[0], 3), 200, dtype=np.uint8)

        valid = np.ones(p.shape[0], dtype=bool)
        if mask is not None:
            m = mask.detach().cpu().numpy().reshape(-1)
            valid &= (m > 0.5) if m.dtype != bool else m
        if conf_list[idx] is not None and conf_thr > 0.0:
            try:
                c = conf_list[idx].detach().cpu().numpy().reshape(-1)
                cmin, cmax = float(c.min()), float(c.max())
                if cmax > cmin:
                    cn = (c - cmin) / (cmax - cmin)
                else:
                    cn = np.ones_like(c)
                valid &= cn >= conf_thr
            except Exception:
                pass

        if valid.any():
            chunks_pts.append(p[valid])
            chunks_rgb.append(rgb_arr[valid])

        f = float(focals[idx]) if idx < len(focals) else max(W, H) * 0.9
        K = np.array([[f, 0.0, W / 2.0], [0.0, f, H / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        intrinsics.append(K)

        c2w = np.asarray(c2w_all[idx], dtype=np.float64).reshape(4, 4)
        try:
            w2c = np.linalg.inv(c2w)
        except np.linalg.LinAlgError:
            w2c = np.eye(4, dtype=np.float64)
        poses_c2w.append(c2w)
        poses_w2c.append(w2c)

    if not chunks_pts:
        raise RuntimeError("DUSt3R produced no 3D points after confidence filtering")

    cloud = np.concatenate(chunks_pts, axis=0).astype(np.float32)
    colors = np.concatenate(chunks_rgb, axis=0).astype(np.uint8)
    if cloud.shape[0] > 500_000:
        idx = np.random.choice(cloud.shape[0], 500_000, replace=False)
        cloud = cloud[idx]
        colors = colors[idx]

    return Dust3rScene(
        points=cloud,
        colors=colors,
        image_paths=[Path(p) for p in masked_image_paths],
        image_sizes=image_sizes,
        intrinsics=intrinsics,
        poses_w2c=poses_w2c,
        poses_c2w=poses_c2w,
    )


def run_dust3r_point_cloud(masked_image_paths: list[Path], settings: RuntimeSettings) -> np.ndarray:
    """Backward-compatible thin wrapper used by the mesh pipeline."""
    scene = run_dust3r_scene(masked_image_paths, settings)
    return scene.points
