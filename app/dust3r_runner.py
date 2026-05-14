from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .runtime_settings import RuntimeSettings


def _torch_safe_globals_ctx():
    """Allow DUSt3R checkpoints that pickle argparse.Namespace on newer torch."""
    try:
        import torch
    except Exception:
        return nullcontext()
    ser = getattr(torch, "serialization", None)
    if ser is None:
        return nullcontext()
    ctx = getattr(ser, "safe_globals", None)
    if callable(ctx):
        try:
            return ctx([argparse.Namespace])
        except Exception:
            return nullcontext()
    add = getattr(ser, "add_safe_globals", None)
    if callable(add):
        try:
            add([argparse.Namespace])
        except Exception:
            pass
    return nullcontext()


def _read_image_resized_rgb(path: Path, width: int, height: int) -> np.ndarray | None:
    """Read ``path`` (color), resize to ``(width, height)``, return uint8 RGB array or None."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    if (img.shape[1], img.shape[0]) != (int(width), int(height)):
        img = cv2.resize(img, (int(width), int(height)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


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
    """Load DUSt3R weights from a local ``.pth`` file, HF hub id, or HF snapshot directory."""

    import os

    from dust3r.model import AsymmetricCroCo3DStereo

    ck_ref = (settings.dust3r_checkpoint_path or "").strip()
    if not ck_ref:
        raise RuntimeError("dust3r_checkpoint_path is empty")

    # Strip accidental quoting from env / JSON on Windows.
    raw = ck_ref.strip().strip('"').strip("'")
    expanded = os.path.expanduser(raw)
    candidates: list[str] = []
    for c in (raw, expanded, os.path.normpath(expanded)):
        if c not in candidates:
            candidates.append(c)
    if not os.path.isabs(expanded):
        abs_path = os.path.abspath(expanded)
        if abs_path not in candidates:
            candidates.append(abs_path)

    local_ckpt: str | None = None
    for cand in candidates:
        # Match DUSt3R upstream: ``os.path.isfile`` (more reliable than Path-only checks on Windows).
        if os.path.isfile(cand):
            local_ckpt = cand
            break

    looks_like_file = raw.lower().endswith((".pth", ".pt"))

    if local_ckpt is not None:
        try:
            from dust3r.model import load_model as dust3r_load_model
        except ImportError as exc:
            raise RuntimeError(
                "DUSt3R is installed but ``load_model`` is missing from dust3r.model — "
                "upgrade naver/dust3r to a recent checkout. "
                f"Original error: {exc}"
            ) from exc
        try:
            with _torch_safe_globals_ctx():
                try:
                    model = dust3r_load_model(local_ckpt, device="cpu", verbose=False)
                except TypeError:
                    model = dust3r_load_model(local_ckpt, device="cpu")
        except Exception as exc:
            msg = str(exc)
            if "Weights only load failed" in msg:
                # PyTorch 2.6+ defaults torch.load(..., weights_only=True). DUSt3R
                # training checkpoints often include argparse.Namespace metadata.
                try:
                    import torch

                    orig_torch_load = torch.load

                    def _torch_load_compat(*args, **kwargs):
                        kwargs.setdefault("weights_only", False)
                        return orig_torch_load(*args, **kwargs)

                    torch.load = _torch_load_compat
                    try:
                        with _torch_safe_globals_ctx():
                            try:
                                model = dust3r_load_model(local_ckpt, device="cpu", verbose=False)
                            except TypeError:
                                model = dust3r_load_model(local_ckpt, device="cpu")
                    finally:
                        torch.load = orig_torch_load
                except Exception as exc2:
                    raise RuntimeError(
                        f"Failed to load DUSt3R weights from local file {local_ckpt!r}. "
                        "Checkpoint likely needs trusted unpickling on this torch version. "
                        f"Original error: {exc2}"
                    ) from exc2
            else:
                raise RuntimeError(
                    f"Failed to load DUSt3R weights from local file {local_ckpt!r}. "
                    "Confirm it is a DUSt3R training checkpoint (``ckpt['args']`` + ``ckpt['model']``). "
                    f"Original error: {exc}"
                ) from exc
        return model.to(device)

    if looks_like_file:
        raise RuntimeError(
            f"DUSt3R checkpoint is not a readable file on this machine: {raw!r}. "
            f"Tried: {candidates}. "
            "Set dust3r_checkpoint_path / POLYGRAPH_OVERRIDE_DUST3R_CHECKPOINT to an existing ``.pth``, "
            "or use a Hugging Face model id (no local path)."
        )

    load_ref = raw

    try:
        model = AsymmetricCroCo3DStereo.from_pretrained(load_ref)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load DUSt3R weights from {load_ref!r}. "
            "For a local file, use an absolute path to a ``.pth`` produced by DUSt3R (see README); "
            "for Hugging Face use a model id (e.g. ``naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt``) "
            "or a snapshot directory. "
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

    max_side = min(int(settings.max_image_side), int(settings.dust3r_max_inference_side))
    images = load_images(paths, size=max_side)
    pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)

    batch_size = 1 if device.type == "cpu" else max(1, int(settings.dust3r_inference_batch_size))
    use_fp16 = device.type == "cuda" and bool(getattr(settings, "dust3r_use_fp16", True))

    amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16) if use_fp16 else nullcontext()

    with amp_ctx:
        output = inference(pairs, model, device, batch_size=batch_size)

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    with amp_ctx:
        scene.compute_global_alignment(
            init="mst",
            niter=int(settings.dust3r_aligner_iters),
            schedule="cosine",
            lr=float(settings.dust3r_aligner_lr),
        )

    pts_list = scene.get_pts3d()
    masks_list = scene.get_masks() if hasattr(scene, "get_masks") else [None] * len(pts_list)
    if hasattr(scene, "get_conf"):
        try:
            conf_list = scene.get_conf()
        except Exception:
            conf_list = [None] * len(pts_list)
    else:
        conf_list = [None] * len(pts_list)

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

    frames: list[dict[str, object]] = []
    image_sizes: list[tuple[int, int]] = []
    intrinsics: list[np.ndarray] = []
    poses_w2c: list[np.ndarray] = []
    poses_c2w: list[np.ndarray] = []

    for idx, (pts, mask) in enumerate(zip(pts_list, masks_list)):
        pts_np = pts.detach().cpu().numpy() if hasattr(pts, "detach") else np.asarray(pts)
        if pts_np.ndim == 3 and pts_np.shape[-1] == 3:
            H, W, _ = pts_np.shape
        else:
            n = int(pts_np.size // 3)
            H = int(np.sqrt(max(1, n)))
            W = int(max(1, n // max(1, H)))
        p = pts_np.reshape(-1, 3)
        image_sizes.append((int(W), int(H)))

        rgb = rgb_list[idx] if idx < len(rgb_list) else None
        rgb_arr = None
        if rgb is not None:
            rgb_np = np.asarray(rgb)
            if rgb_np.dtype == np.uint8:
                rgb_arr = rgb_np.reshape(-1, 3)
            else:
                f = rgb_np.astype(np.float32)
                lo, hi = float(f.min()), float(f.max())
                if lo < -0.05 or hi > 1.05:
                    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                    f = f * std + mean
                rgb_arr = (np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8).reshape(-1, 3)

        if (
            rgb_arr is None
            or rgb_arr.shape[0] != p.shape[0]
            or int(np.asarray(rgb_arr).max()) <= 4
        ):
            disk = _read_image_resized_rgb(masked_image_paths[idx], W, H)
            if disk is not None:
                rgb_arr = disk.reshape(-1, 3)
            elif rgb_arr is None:
                rgb_arr = np.full((p.shape[0], 3), 200, dtype=np.uint8)

        cidx = conf_list[idx] if idx < len(conf_list) else None
        frames.append({"p": p, "rgb_arr": rgb_arr, "mask": mask, "conf": cidx})

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

    def _merge_chunks(conf_use: float, use_mask: bool) -> tuple[list[np.ndarray], list[np.ndarray]]:
        out_pts: list[np.ndarray] = []
        out_rgb: list[np.ndarray] = []
        for frame in frames:
            p = frame["p"]  # type: ignore[assignment]
            rgb_arr = frame["rgb_arr"]  # type: ignore[assignment]
            mask = frame["mask"] if use_mask else None  # type: ignore[assignment]
            conf_t = frame["conf"]  # type: ignore[assignment]

            valid = np.ones(p.shape[0], dtype=bool)
            if mask is not None:
                m = mask.detach().cpu().numpy().reshape(-1)  # type: ignore[union-attr]
                valid &= (m > 0.5) if m.dtype != bool else m
            if conf_t is not None and conf_use > 0.0:
                try:
                    c = conf_t.detach().cpu().numpy().reshape(-1)  # type: ignore[union-attr]
                    cmin, cmax = float(c.min()), float(c.max())
                    if cmax > cmin:
                        cn = (c - cmin) / (cmax - cmin)
                    else:
                        cn = np.ones_like(c)
                    valid &= cn >= conf_use
                except Exception:
                    pass
            if valid.any():
                out_pts.append(p[valid])
                out_rgb.append(rgb_arr[valid])
        return out_pts, out_rgb

    chunks_pts, chunks_rgb = _merge_chunks(conf_thr, True)
    tried: list[str] = [f"threshold={conf_thr}, mask=on"]
    if not chunks_pts and conf_thr > 0.0:
        chunks_pts, chunks_rgb = _merge_chunks(0.0, True)
        tried.append("threshold=0, mask=on")
    if not chunks_pts:
        chunks_pts, chunks_rgb = _merge_chunks(0.0, False)
        tried.append("threshold=0, mask=off")

    if not chunks_pts:
        raise RuntimeError(
            "DUSt3R produced no 3D points after filtering. "
            f"Tried: {'; '.join(tried)}. "
            "Set ``dust3r_confidence_threshold`` to 0 in PUT /settings, "
            "try ``sam_segmentation_mode`` / more overlapping views, "
            "or switch ``reconstruction_backend`` to ``colmap`` for this scene."
        )

    cloud = np.concatenate(chunks_pts, axis=0).astype(np.float32)
    colors = np.concatenate(chunks_rgb, axis=0).astype(np.uint8)
    if cloud.shape[0] > 500_000:
        idx = np.random.choice(cloud.shape[0], 500_000, replace=False)
        cloud = cloud[idx]
        colors = colors[idx]

    # Free DUSt3R weights + optimizer state before GS train.py (same worker process shares CUDA).
    try:
        del frames
        del pts_list
        del masks_list
        del conf_list
        del rgb_list
        del output
        del pairs
        del images
        del model
        del scene
    except Exception:
        pass
    try:
        from .cuda_memory import purge_torch_cuda

        purge_torch_cuda()
    except Exception:
        pass

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
