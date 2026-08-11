"""Mask generation from studio cutout (after) images."""

from __future__ import annotations

import cv2
import numpy as np


def _read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None or img.size == 0:
        raise ValueError(f"Could not read image: {path}")
    return img


def mask_from_after_image(*, before_shape: tuple[int, int], after_bgr_or_bgra: np.ndarray) -> np.ndarray:
    """Derive binary foreground mask (0/255 uint8) from an after (studio cutout) image."""
    h, w = before_shape
    img = after_bgr_or_bgra
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[0] != h or img.shape[1] != w:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    if img.shape[2] >= 4:
        alpha = img[:, :, 3].astype(np.uint8)
        fg = alpha >= 128
    else:
        fg = _foreground_from_studio_cutout(img[:, :, :3])

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[fg] = 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask


def _foreground_from_studio_cutout(bgr: np.ndarray) -> np.ndarray:
    """Foreground = not the connected white/low-chroma studio background.

    Corner flood-fill keeps product color (even light beige) and follows shape
    better than a hard RGB>=245 threshold.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    rgb = bgr.astype(np.int16)
    near_white = (sat <= 30) & (val >= 225)
    near_white |= (rgb[:, :, 0] >= 242) & (rgb[:, :, 1] >= 242) & (rgb[:, :, 2] >= 242)

    hh, ww = near_white.shape
    bg = np.zeros((hh, ww), dtype=bool)
    stack = []
    for y, x in ((0, 0), (0, ww - 1), (hh - 1, 0), (hh - 1, ww - 1)):
        if near_white[y, x]:
            stack.append((y, x))
    while stack:
        y, x = stack.pop()
        if bg[y, x] or not near_white[y, x]:
            continue
        bg[y, x] = True
        if y > 0:
            stack.append((y - 1, x))
        if y + 1 < hh:
            stack.append((y + 1, x))
        if x > 0:
            stack.append((y, x - 1))
        if x + 1 < ww:
            stack.append((y, x + 1))

    fg = ~bg
    if int(fg.sum()) < (hh * ww) * 0.002:
        fg = ~near_white
    return fg


def generate_and_save_mask(*, before_path: str, after_path: str, mask_path: str) -> None:
    before = _read_image(before_path)
    after = _read_image(after_path)
    h, w = before.shape[:2]
    mask = mask_from_after_image(before_shape=(h, w), after_bgr_or_bgra=after)
    cv2.imwrite(mask_path, mask)


def refine_mask(mask: np.ndarray, *, open_k: int = 3, close_k: int = 5) -> np.ndarray:
    """Light morphology cleanup on predicted mask."""
    if mask.ndim != 2:
        raise ValueError("mask must be 2D grayscale")
    out = mask.astype(np.uint8)
    if open_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    return out
