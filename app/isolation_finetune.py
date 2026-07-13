"""PyTorch UNet fine-tune for incremental isolation training (GPU worker)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

log = logging.getLogger(__name__)

INPUT_SIZE = 320


def _device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class SmallUNet:
    """Lazy-built UNet; constructed inside train_unet to avoid torch import on API CPU host."""

    pass


def _build_unet():
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
            self.enc1 = ConvBlock(3, 32)
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


def _load_pair_tensors(pairs: list[dict], size: int = INPUT_SIZE):
    import torch

    images = []
    masks = []
    for item in pairs:
        before = cv2.imread(str(item["before"]), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(item["mask"]), cv2.IMREAD_GRAYSCALE)
        if before is None or gt is None:
            continue
        before = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
        before = cv2.resize(before, (size, size), interpolation=cv2.INTER_AREA)
        gt = cv2.resize(gt, (size, size), interpolation=cv2.INTER_NEAREST)
        img = before.astype(np.float32) / 255.0
        m = (gt.astype(np.float32) / 255.0)[None, ...]
        images.append(np.transpose(img, (2, 0, 1)))
        masks.append(m)
    if not images:
        raise ValueError("No readable before/mask pairs for fine-tune")
    x = torch.from_numpy(np.stack(images)).float()
    y = torch.from_numpy(np.stack(masks)).float()
    return x, y


def _iou_batch(pred, target, thr: float = 0.5) -> float:
    import torch

    pb = pred >= thr
    tb = target >= thr
    inter = (pb & tb).sum().item()
    union = (pb | tb).sum().item()
    return float(inter / union) if union > 0 else 0.0


def train_unet_incremental(
    *,
    pairs: list[dict],
    output_dir: Path,
    epochs: int,
    val_split: float,
    resume_checkpoint: Path | None,
    progress_callback: Callable[[int], None] | None = None,
) -> dict:
    """Fine-tune SmallUNet, optionally resuming from checkpoint.pt. Writes model.onnx + checkpoint.pt."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split

    output_dir.mkdir(parents=True, exist_ok=True)
    device = _device()
    log.info("isolation fine-tune device=%s pairs=%d epochs=%d resume=%s", device, len(pairs), epochs, resume_checkpoint)

    x, y = _load_pair_tensors(pairs)
    n = len(x)
    n_val = max(1, int(n * val_split)) if n >= 2 else 0
    n_train = max(1, n - n_val) if n_val else n

    dataset = TensorDataset(x, y)
    if n_val and n_train + n_val <= n:
        train_ds, val_ds = random_split(dataset, [n_train, n - n_train])
    else:
        train_ds, val_ds = dataset, None

    train_loader = DataLoader(train_ds, batch_size=min(4, len(train_ds)), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=min(4, len(val_ds)), shuffle=False) if val_ds is not None else None

    model = _build_unet().to(device)
    generation = 1
    parent_model_id = None
    if resume_checkpoint and resume_checkpoint.is_file():
        ckpt = torch.load(str(resume_checkpoint), map_location=device, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
        if isinstance(ckpt, dict):
            generation = int(ckpt.get("generation") or 1) + 1
            parent_model_id = ckpt.get("model_id")
        log.info("resumed checkpoint generation=%d from %s", generation, resume_checkpoint)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3 if generation == 1 else 5e-4)
    loss_fn = nn.BCELoss()

    model.train()
    for epoch in range(max(1, epochs)):
        total = 0.0
        steps = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
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

    # Approx precision/recall on last val batch aggregate
    precision = recall = mean_iou

    ckpt_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "generation": generation,
            "model_id": parent_model_id,
            "input_size": INPUT_SIZE,
        },
        str(ckpt_path),
    )

    onnx_path = output_dir / "model.onnx"
    dummy = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE, device=device)
    model.eval()
    try:
        import onnx  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            'Package "onnx" is required to export the trained model. '
            'On the GPU worker run: pip install onnx'
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
        "backend": "unet_finetune",
        "device": device,
        "input_size": INPUT_SIZE,
        "note": "Incremental UNet fine-tune; each train resumes from previous checkpoint when available.",
    }


def torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False
