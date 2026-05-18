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
    """Aggregated DUSt3R global-aligner output for meshing / camera-aware color bake."""

    points: np.ndarray
    colors: np.ndarray
    image_paths: list[Path] = field(default_factory=list)
    image_sizes: list[tuple[int, int]] = field(default_factory=list)
    intrinsics: list[np.ndarray] = field(default_factory=list)
    poses_w2c: list[np.ndarray] = field(default_factory=list)
    poses_c2w: list[np.ndarray] = field(default_factory=list)


def _resolve_device(settings: RuntimeSettings):
    import torch

    device_str = settings.device
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _subsample_image_paths(paths: list[Path], max_n: int) -> list[Path]:
    if len(paths) <= max_n:
        return list(paths)
    idx = np.linspace(0, len(paths) - 1, max_n)
    idx = np.unique(np.round(idx).astype(int))
    out = [paths[int(i)] for i in idx]
    if len(out) < 2:
        return [paths[0], paths[-1]]
    return out


def _resolve_dust3r_scene_graph(settings: RuntimeSettings, n_views: int) -> str:
    raw = (getattr(settings, "dust3r_scene_graph", None) or "auto").strip()
    if raw.lower() != "auto":
        return raw
    cap = int(getattr(settings, "dust3r_complete_graph_max_views", 24))
    if n_views <= cap:
        return "complete"
    return "swin-6-noncyclic"


def _load_model(settings: RuntimeSettings, device):
    import os
    from dust3r.model import AsymmetricCroCo3DStereo

    ck_ref = (settings.dust3r_checkpoint_path or "").strip()
    if not ck_ref:
        raise RuntimeError("dust3r_checkpoint_path is empty")

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
        if os.path.isfile(cand):
            local_ckpt = cand
            break

    looks_like_file = raw.lower().endswith((".pth", ".pt"))

    if local_ckpt is not None:
        try:
            from dust3r.model import load_model as dust3r_load_model
        except ImportError as exc:
            raise RuntimeError("dust3r.model.load_model missing; update naver/dust3r checkout.") from exc
        try:
            with _torch_safe_globals_ctx():
                try:
                    model = dust3r_load_model(local_ckpt, device="cpu", verbose=False)
                except TypeError:
                    model = dust3r_load_model(local_ckpt, device="cpu")
        except Exception as exc:
            msg = str(exc)
            if "Weights only load failed" in msg:
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
                    raise RuntimeError(f"Failed loading DUSt3R local checkpoint {local_ckpt!r}: {exc2}") from exc2
            else:
                raise RuntimeError(f"Failed loading DUSt3R local checkpoint {local_ckpt!r}: {exc}") from exc
        return model.to(device)

    if looks_like_file:
        raise RuntimeError(
            f"DUSt3R checkpoint file not found on this machine: {raw!r}. Tried: {candidates}. "
            "Set dust3r_checkpoint_path or POLYGRAPH_OVERRIDE_DUST3R_CHECKPOINT to a valid .pth path."
        )

    load_ref = raw
    try:
        model = AsymmetricCroCo3DStereo.from_pretrained(load_ref)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load DUSt3R weights from {load_ref!r}. "
            "Use local .pth path or HF model id (e.g. naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt)."
        ) from exc
    return model.to(device)


def run_dust3r_scene(masked_image_paths: list[Path], settings: RuntimeSettings) -> Dust3rScene:
    if len(masked_image_paths) < 2:
        raise ValueError("DUSt3R needs at least 2 images")

    import torch

    try:
        from dust3r.cloud_opt import GlobalAlignerMode, global_aligner
        from dust3r.image_pairs import make_pairs
        from dust3r.inference import inference
        from dust3r.utils.image import load_images
    except ImportError as exc:
        raise RuntimeError(
            "DUSt3R package is not installed. Install naver/dust3r in this environment."
        ) from exc

    max_views = int(getattr(settings, "dust3r_max_input_views", 36))
    working_paths = _subsample_image_paths(list(masked_image_paths), max_views)
    scene_graph = _resolve_dust3r_scene_graph(settings, len(working_paths))

    paths = [str(p.resolve()) for p in working_paths]
    device = _resolve_device(settings)
    model = _load_model(settings, device)

    max_side = min(int(settings.max_image_side), int(settings.dust3r_max_inference_side))
    images = load_images(paths, size=max_side)
    pairs = make_pairs(images, scene_graph=scene_graph, prefilter=None, symmetrize=True)

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

    try:
        focals = scene.get_focals().detach().cpu().numpy().reshape(-1)
    except Exception:
        focals = np.zeros(len(pts_list), dtype=np.float32)
    try:
        c2w_all = scene.get_im_poses().detach().cpu().numpy()
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
            h, w, _ = pts_np.shape
        else:
            n = int(pts_np.size // 3)
            h = int(np.sqrt(max(1, n)))
            w = int(max(1, n // max(1, h)))
        p = pts_np.reshape(-1, 3)
        image_sizes.append((int(w), int(h)))

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
        if rgb_arr is None or rgb_arr.shape[0] != p.shape[0] or int(np.asarray(rgb_arr).max()) <= 4:
            disk = _read_image_resized_rgb(working_paths[idx], w, h)
            if disk is not None:
                rgb_arr = disk.reshape(-1, 3)
            elif rgb_arr is None:
                rgb_arr = np.full((p.shape[0], 3), 200, dtype=np.uint8)

        cidx = conf_list[idx] if idx < len(conf_list) else None
        frames.append({"p": p, "rgb_arr": rgb_arr, "mask": mask, "conf": cidx})

        f = float(focals[idx]) if idx < len(focals) else max(w, h) * 0.9
        K = np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)
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
                    cn = (c - cmin) / (cmax - cmin) if cmax > cmin else np.ones_like(c)
                    valid &= cn >= conf_use
                except Exception:
                    pass
            if valid.any():
                out_pts.append(p[valid])
                out_rgb.append(rgb_arr[valid])
        return out_pts, out_rgb

    chunks_pts, chunks_rgb = _merge_chunks(conf_thr, True)
    if not chunks_pts and conf_thr > 0.0:
        chunks_pts, chunks_rgb = _merge_chunks(0.0, True)
    if not chunks_pts:
        chunks_pts, chunks_rgb = _merge_chunks(0.0, False)
    if not chunks_pts:
        raise RuntimeError(
            "DUSt3R produced no 3D points after filtering. "
            "Set dust3r_confidence_threshold=0, add more overlap, or switch backend."
        )

    cloud = np.concatenate(chunks_pts, axis=0).astype(np.float32)
    colors = np.concatenate(chunks_rgb, axis=0).astype(np.uint8)
    if cloud.shape[0] > 500_000:
        idx = np.random.choice(cloud.shape[0], 500_000, replace=False)
        cloud = cloud[idx]
        colors = colors[idx]

    try:
        from .cuda_memory import purge_torch_cuda

        purge_torch_cuda()
    except Exception:
        pass

    return Dust3rScene(
        points=cloud,
        colors=colors,
        image_paths=[Path(p) for p in working_paths],
        image_sizes=image_sizes,
        intrinsics=intrinsics,
        poses_w2c=poses_w2c,
        poses_c2w=poses_c2w,
    )

