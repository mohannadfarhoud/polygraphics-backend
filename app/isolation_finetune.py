"""PyTorch UNet fine-tune for incremental isolation training (GPU worker).

Training bias: color-edge / boundary patterns over absolute size & position.
- Strong scale/translate/crop augmentations (break position/size memorization)
- Edge-channel input (RGB + color-edge magnitude)
- Boundary-weighted loss so mask borders dominate the objective
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

log = logging.getLogger(__name__)

INPUT_SIZE = 320
# Loss mix: focus on edges/boundaries more than bulk region fill.
EDGE_LOSS_WEIGHT = float(os.getenv("ISOLATION_EDGE_LOSS_WEIGHT", "2.5"))
BCE_WEIGHT = float(os.getenv("ISOLATION_BCE_WEIGHT", "0.35"))
DICE_WEIGHT = float(os.getenv("ISOLATION_DICE_WEIGHT", "0.65"))


def _device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def color_edge_map(rgb: np.ndarray) -> np.ndarray:
    """Color-aware edge magnitude in [0,1] (float32 HxW). Emphasizes hue/chroma boundaries."""
    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2LAB)
    # Edges on L (structure) + a/b (color) — color channels weighted higher.
    edges = []
    weights = (0.35, 0.9, 0.9)  # L, a, b
    for i, w in enumerate(weights):
        ch = lab[:, :, i]
        gx = cv2.Sobel(ch, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(ch, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        edges.append(w * mag)
    edge = np.maximum.reduce(edges)
    # Soft normalize per-image so absolute contrast doesn't dominate.
    p95 = float(np.percentile(edge, 95)) + 1e-6
    edge = np.clip(edge / p95, 0.0, 1.0).astype(np.float32)
    return edge


def pack_input_chw(rgb: np.ndarray) -> np.ndarray:
    """RGB uint8/float HxWx3 -> float32 CHW with 4 channels (RGB + color-edge)."""
    if rgb.dtype != np.uint8:
        rgb_u8 = np.clip(rgb * 255.0 if rgb.max() <= 1.5 else rgb, 0, 255).astype(np.uint8)
    else:
        rgb_u8 = rgb
    edge = color_edge_map(rgb_u8)
    img = rgb_u8.astype(np.float32) / 255.0
    stacked = np.concatenate([img, edge[:, :, None]], axis=2)  # HWC 4
    return np.transpose(stacked, (2, 0, 1)).astype(np.float32)


def _build_unet(in_ch: int = 4):
    import torch
    import torch.nn as nn

    class ConvBlock(nn.Module):
        def __init__(self, cin: int, cout: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.net(x)

    class UNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc1 = ConvBlock(in_ch, 32)
            self.enc2 = ConvBlock(32, 64)
            self.enc3 = ConvBlock(64, 128)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = ConvBlock(128, 256)
            self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.dec3 = ConvBlock(256, 128)
            self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dec2 = ConvBlock(128, 64)
            self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dec1 = ConvBlock(64, 32)
            self.out = nn.Conv2d(32, 1, 1)

        def forward(self, x):  # type: ignore[no-untyped-def]
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))
            d3 = self.up3(b)
            d3 = self.dec3(torch.cat([d3, e3], dim=1))
            d2 = self.up2(d3)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d1 = self.up1(d2)
            d1 = self.dec1(torch.cat([d1, e1], dim=1))
            return torch.sigmoid(self.out(d1))

    return UNet()


def _geom_augment(rgb: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Random scale + translate + crop/pad so the net cannot rely on fixed size/position."""
    h, w = rgb.shape[:2]
    # Scale heavily so absolute object size is unreliable.
    scale = random.uniform(0.55, 1.45)
    nh, nw = max(8, int(h * scale)), max(8, int(w * scale))
    rgb_s = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask_s = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

    # Place onto canvas with random offset (translation invariance).
    canvas_rgb = np.full((size, size, 3), 255, dtype=np.uint8)  # white-ish studio bias
    canvas_m = np.zeros((size, size), dtype=np.uint8)
    # If scaled larger than canvas, random crop; else random paste.
    if nh >= size and nw >= size:
        y0 = random.randint(0, nh - size)
        x0 = random.randint(0, nw - size)
        canvas_rgb = rgb_s[y0 : y0 + size, x0 : x0 + size]
        canvas_m = mask_s[y0 : y0 + size, x0 : x0 + size]
    else:
        max_y = max(0, size - nh)
        max_x = max(0, size - nw)
        y0 = random.randint(0, max_y) if max_y > 0 else 0
        x0 = random.randint(0, max_x) if max_x > 0 else 0
        y1, x1 = min(size, y0 + nh), min(size, x0 + nw)
        canvas_rgb[y0:y1, x0:x1] = rgb_s[: y1 - y0, : x1 - x0]
        canvas_m[y0:y1, x0:x1] = mask_s[: y1 - y0, : x1 - x0]

    if random.random() < 0.5:
        canvas_rgb = cv2.flip(canvas_rgb, 1)
        canvas_m = cv2.flip(canvas_m, 1)

    # Mild photometric jitter — keep color relationships, avoid destroying edges.
    if random.random() < 0.7:
        alpha = random.uniform(0.85, 1.15)  # contrast
        beta = random.uniform(-12, 12)  # brightness
        canvas_rgb = np.clip(canvas_rgb.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if random.random() < 0.4:
        # Slight hue shift in HSV (color identity still mostly intact).
        hsv = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.randint(-8, 8)) % 180
        canvas_rgb = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)

    return canvas_rgb, canvas_m


def _load_raw_pairs(pairs: list[dict]) -> list[tuple[np.ndarray, np.ndarray]]:
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for item in pairs:
        before = cv2.imread(str(item["before"]), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(item["mask"]), cv2.IMREAD_GRAYSCALE)
        if before is None or gt is None:
            continue
        before = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
        out.append((before, gt))
    if not out:
        raise ValueError("No readable before/mask pairs for fine-tune")
    return out


def _mask_boundary_weight(mask_t, device):
    """Higher loss weight on GT mask boundaries (color-edge / silhouette)."""
    import torch
    import torch.nn.functional as F

    # Soft Sobel on mask → boundary band.
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
    gx = F.conv2d(mask_t, kx, padding=1)
    gy = F.conv2d(mask_t, ky, padding=1)
    edge = torch.sqrt(gx * gx + gy * gy + 1e-6)
    edge = edge / (edge.amax(dim=(2, 3), keepdim=True) + 1e-6)
    # Base 1.0 everywhere, boost boundaries.
    return 1.0 + EDGE_LOSS_WEIGHT * edge


def _dice_loss(pred, target, eps: float = 1e-6):
    import torch

    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def _edge_consistency_loss(pred, target):
    """Match predicted silhouette edges to GT edges (color-edge focus)."""
    import torch
    import torch.nn.functional as F

    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
    pe = torch.sqrt(F.conv2d(pred, kx, padding=1) ** 2 + F.conv2d(pred, ky, padding=1) ** 2 + 1e-6)
    te = torch.sqrt(F.conv2d(target, kx, padding=1) ** 2 + F.conv2d(target, ky, padding=1) ** 2 + 1e-6)
    pe = pe / (pe.amax(dim=(2, 3), keepdim=True) + 1e-6)
    te = te / (te.amax(dim=(2, 3), keepdim=True) + 1e-6)
    return F.l1_loss(pe, te)


def _iou_batch(pred, target, thr: float = 0.5) -> float:
    pb = pred >= thr
    tb = target >= thr
    inter = (pb & tb).sum().item()
    union = (pb | tb).sum().item()
    return float(inter / union) if union > 0 else 0.0


class _PairDataset:
    """On-the-fly geometric + color-edge packing (train) or center resize (val)."""

    def __init__(self, pairs: list[tuple[np.ndarray, np.ndarray]], *, augment: bool, size: int = INPUT_SIZE):
        self.pairs = pairs
        self.augment = augment
        self.size = size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        import torch

        rgb, mask = self.pairs[idx]
        if self.augment:
            rgb, mask = _geom_augment(rgb, mask, self.size)
        else:
            rgb = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        x = pack_input_chw(rgb)
        y = (mask.astype(np.float32) / 255.0)[None, ...]
        return torch.from_numpy(x), torch.from_numpy(y)


def train_unet_incremental(
    *,
    pairs: list[dict],
    output_dir: Path,
    epochs: int,
    val_split: float,
    resume_checkpoint: Path | None,
    progress_callback: Callable[[int], None] | None = None,
) -> dict:
    """Fine-tune UNet focused on color edges; geometric aug breaks size/position bias."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, random_split

    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device()
    raw = _load_raw_pairs(pairs)
    n = len(raw)
    n_val = max(1, int(n * val_split)) if n >= 2 else 0
    n_train = max(1, n - n_val) if n_val else n

    full_idx = list(range(n))
    random.shuffle(full_idx)
    val_idx = set(full_idx[:n_val]) if n_val else set()
    train_raw = [raw[i] for i in range(n) if i not in val_idx] or raw
    val_raw = [raw[i] for i in val_idx] if val_idx else None

    train_ds = _PairDataset(train_raw, augment=True)
    val_ds = _PairDataset(val_raw, augment=False) if val_raw else None

    train_loader = DataLoader(train_ds, batch_size=min(4, len(train_ds)), shuffle=True, num_workers=0)
    val_loader = (
        DataLoader(val_ds, batch_size=min(4, len(val_ds)), shuffle=False, num_workers=0) if val_ds else None
    )

    model = _build_unet(in_ch=4).to(device)
    generation = 1
    parent_model_id = None
    if resume_checkpoint and resume_checkpoint.is_file():
        ckpt = torch.load(str(resume_checkpoint), map_location=device, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            log.warning(
                "checkpoint partial load (likely 3ch→4ch upgrade): missing=%s unexpected=%s",
                incompatible.missing_keys[:4],
                incompatible.unexpected_keys[:4],
            )
        if isinstance(ckpt, dict):
            generation = int(ckpt.get("generation") or 1) + 1
            parent_model_id = ckpt.get("model_id")
        log.info("resumed checkpoint generation=%d from %s", generation, resume_checkpoint)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3 if generation == 1 else 5e-4)

    log.info(
        "isolation fine-tune edge-focus device=%s pairs=%d epochs=%d edge_w=%.2f",
        device,
        n,
        epochs,
        EDGE_LOSS_WEIGHT,
    )

    model.train()
    for epoch in range(max(1, epochs)):
        total = 0.0
        steps = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            weights = _mask_boundary_weight(yb, device)
            bce = F.binary_cross_entropy(pred, yb, weight=weights)
            dice = _dice_loss(pred, yb)
            edge_l = _edge_consistency_loss(pred, yb)
            loss = BCE_WEIGHT * bce + DICE_WEIGHT * dice + EDGE_LOSS_WEIGHT * 0.5 * edge_l
            loss.backward()
            opt.step()
            total += float(loss.item())
            steps += 1
        if progress_callback:
            pct = 20 + int(70 * (epoch + 1) / max(1, epochs))
            progress_callback(min(90, pct))
        log.info("epoch %d/%d loss=%.4f", epoch + 1, epochs, total / max(1, steps))

    model.eval()
    ious: list[float] = []
    with torch.no_grad():
        loader = val_loader or train_loader
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            ious.append(_iou_batch(pred, yb))
    mean_iou = float(np.mean(ious)) if ious else 0.0
    precision = recall = mean_iou

    ckpt_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "generation": generation,
            "model_id": parent_model_id,
            "input_size": INPUT_SIZE,
            "in_channels": 4,
            "focus": "color_edge",
        },
        str(ckpt_path),
    )

    onnx_path = output_dir / "model.onnx"
    dummy = torch.randn(1, 4, INPUT_SIZE, INPUT_SIZE, device=device)
    model.eval()
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            'Package "onnx" is required to export the trained model. '
            "On the GPU worker run: pip install onnx"
        ) from exc
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["mask"],
        dynamic_axes={"input": {0: "batch"}, "mask": {0: "batch"}},
        opset_version=17,
    )

    return {
        "iou": round(mean_iou, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "val_pairs": n_val or 0,
        "train_pairs": n_train,
        "epochs": epochs,
        "generation": generation,
        "parent_model_id": parent_model_id,
        "backend": "unet_finetune_color_edge",
        "device": device,
        "input_size": INPUT_SIZE,
        "in_channels": 4,
        "edge_loss_weight": EDGE_LOSS_WEIGHT,
        "note": (
            "Incremental UNet biased to color-edge/boundary patterns; "
            "strong scale/translate aug reduces size/position memorization."
        ),
    }


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False
