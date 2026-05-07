from __future__ import annotations

from pathlib import Path

import numpy as np

from .runtime_settings import RuntimeSettings


def run_dust3r_point_cloud(masked_image_paths: list[Path], settings: RuntimeSettings) -> np.ndarray:
    """
    Run DUSt3R + global alignment and return Nx3 float32 points in a unified frame.
    Requires: torch, torchvision, and the naver/dust3r package installed from source.
    """
    try:
        import torch
        from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
        from dust3r.image_pairs import make_pairs
        from dust3r.inference import inference
        from dust3r.model import AsymmetricCroCo3DStereo
        from dust3r.utils.image import load_images
    except ImportError as exc:
        raise RuntimeError(
            "DUSt3R Python package is not installed. Clone https://github.com/naver/dust3r "
            "and install it in this environment (pip install -e .), plus torch/torchvision. "
            "Original error: "
            + str(exc)
        ) from exc

    paths = [str(p.resolve()) for p in masked_image_paths]
    device_str = settings.device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

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
    model = model.to(device)

    max_side = min(settings.max_image_side, 512)
    images = load_images(paths, size=max_side)
    pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)

    batch_size = 1 if device.type == "cpu" else 2
    output = inference(pairs, model, device, batch_size=batch_size)

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.compute_global_alignment(init="mst", niter=300, schedule="cosine", lr=0.01)

    pts_list = scene.get_pts3d()
    masks_list = scene.get_masks() if hasattr(scene, "get_masks") else [None] * len(pts_list)

    chunks: list[np.ndarray] = []
    for pts, mask in zip(pts_list, masks_list):
        p = pts.detach().cpu().numpy().reshape(-1, 3)
        if mask is None:
            chunks.append(p)
            continue
        m = mask.detach().cpu().numpy().reshape(-1)
        if m.dtype != bool:
            m = m > 0.5
        chunks.append(p[m])

    if not chunks:
        raise RuntimeError("DUSt3R produced no 3D points")

    cloud = np.concatenate(chunks, axis=0).astype(np.float32)
    if cloud.shape[0] > 500_000:
        idx = np.random.choice(cloud.shape[0], 500_000, replace=False)
        cloud = cloud[idx]
    return cloud
