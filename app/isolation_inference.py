"""ONNX Runtime CPU inference for isolation/matting models."""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .isolation_mask import refine_mask

log = logging.getLogger(__name__)

_SESSION_LOCK = threading.Lock()
_SESSION: object | None = None
_SESSION_MODEL_PATH: str | None = None


def onnxruntime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401

        return True
    except Exception:
        return False


def _load_session(model_path: Path):
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = int(os.getenv("ISOLATION_ORT_THREADS", "2"))
    opts.intra_op_num_threads = int(os.getenv("ISOLATION_ORT_THREADS", "2"))
    return ort.InferenceSession(str(model_path), sess_options=opts, providers=providers)


def get_or_load_session(model_path: Path):
    global _SESSION, _SESSION_MODEL_PATH
    key = str(model_path.resolve())
    with _SESSION_LOCK:
        if _SESSION is not None and _SESSION_MODEL_PATH == key:
            return _SESSION
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        log.info("loading isolation ONNX model: %s", key)
        _SESSION = _load_session(model_path)
        _SESSION_MODEL_PATH = key
        return _SESSION


def invalidate_session() -> None:
    global _SESSION, _SESSION_MODEL_PATH
    with _SESSION_LOCK:
        _SESSION = None
        _SESSION_MODEL_PATH = None


def _session_input_size(session) -> int:
    """Infer spatial input size from ONNX graph (rembg ISNet uses 1024)."""
    try:
        shape = session.get_inputs()[0].shape
        # typical: [1, 3, H, W] or [batch, 3, H, W]
        if len(shape) >= 4:
            h, w = shape[-2], shape[-1]
            if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
                return int(max(h, w))
    except Exception:
        pass
    env = os.getenv("ISOLATION_INPUT_SIZE", "").strip()
    if env.isdigit():
        return int(env)
    return 1024


def _preprocess_isnet(image_bgr: np.ndarray, size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Match rembg DisSession / isnet-general-use normalization."""
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    im_ary = resized.astype(np.float32)
    im_ary = im_ary / max(float(np.max(im_ary)), 1e-6)
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    tmp = (im_ary - mean) / std
    arr = np.transpose(tmp, (2, 0, 1))[None, ...].astype(np.float32)
    return arr, (h, w)


def _preprocess_simple(image_bgr: np.ndarray, size: int = 320) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...]
    return arr, (h, w)


def _session_input_channels(session) -> int:
    try:
        shape = session.get_inputs()[0].shape
        if len(shape) >= 2:
            c = shape[1]
            if isinstance(c, int) and c > 0:
                return int(c)
    except Exception:
        pass
    return 3


def _preprocess_color_edge(image_bgr: np.ndarray, size: int = 512) -> tuple[np.ndarray, tuple[int, int]]:
    """4-channel RGB + color-edge input (matches isolation_finetune UNet)."""
    from .isolation_finetune import pack_input_chw

    h, w = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    chw = pack_input_chw(resized)
    return chw[None, ...], (h, w)


def _postprocess_mask(output: np.ndarray, original_hw: tuple[int, int], *, minmax: bool) -> np.ndarray:
    h, w = original_hw
    if output.ndim == 4:
        matte = output[0, 0]
    elif output.ndim == 3:
        matte = output[0]
    else:
        matte = output
    matte = matte.astype(np.float32)
    if minmax:
        ma = float(np.max(matte))
        mi = float(np.min(matte))
        matte = (matte - mi) / (ma - mi + 1e-8)
    else:
        matte = np.clip(matte, 0.0, 1.0)
    mask = (matte * 255.0).astype(np.uint8)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary


def predict_mask(image_bgr: np.ndarray, *, model_path: Path) -> np.ndarray:
    session = get_or_load_session(model_path)
    size = _session_input_size(session)
    channels = _session_input_channels(session)

    if channels >= 4:
        # Growing UNet: RGB + color-edge channel
        if not isinstance(size, int) or size <= 0 or size > 2048:
            size = 512
        inp, hw = _preprocess_color_edge(image_bgr, size=size)
        use_isnet = False
    elif size >= 512 or os.getenv("ISOLATION_PREPROCESS", "").strip().lower() in (
        "isnet",
        "rembg",
        "dis",
    ):
        inp, hw = _preprocess_isnet(image_bgr, size=size)
        use_isnet = True
    else:
        inp, hw = _preprocess_simple(image_bgr, size=size if size > 0 else 320)
        use_isnet = False

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: inp})
    if not outputs:
        raise RuntimeError("ONNX session returned no outputs")
    mask = _postprocess_mask(outputs[0], hw, minmax=use_isnet)
    if os.getenv("ISOLATION_MORPH_CLEANUP", "1").strip().lower() not in ("0", "false", "no"):
        mask = refine_mask(mask)
    return mask


def composite_rgba(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bgra = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = mask
    return bgra


def predict_isolated_png(image_bytes: bytes, *, model_path: Path) -> tuple[bytes, bytes, int]:
    """Return (isolated_rgba_png, mask_png, latency_ms)."""
    t0 = time.perf_counter()
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Invalid image file")
    mask = predict_mask(image_bgr, model_path=model_path)
    rgba = composite_rgba(image_bgr, mask)
    ok1, enc_rgba = cv2.imencode(".png", rgba)
    ok2, enc_mask = cv2.imencode(".png", mask)
    if not ok1 or not ok2:
        raise RuntimeError("Failed to encode PNG output")
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return enc_rgba.tobytes(), enc_mask.tobytes(), latency_ms
