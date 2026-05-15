#!/usr/bin/env python3
"""Generate binary foreground masks with Segment Anything on CUDA.

Reads images from --input-dir, writes PNG masks (--mask-dir) and optionally
masked RGB copies (--masked-dir, black background).

Requires: pip install segment-anything torch torchvision opencv-python

Example (RTX GPU):

    .venv\\Scripts\\python.exe scripts\\sam_masks_cuda.py ^
      --input-dir C:\\shots\\in ^
      --mask-dir C:\\shots\\masks ^
      --masked-dir C:\\shots\\masked ^
      --checkpoint C:\\polyGraphics\\models\\sam\\sam_vit_b_01ec64.pth ^
      --model-type vit_b
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.segmentation import refine_binary_mask_to_center_subject


def main() -> int:
    parser = argparse.ArgumentParser(description="SAM binary masks on CUDA")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--mask-dir", type=Path, required=True)
    parser.add_argument("--masked-dir", type=Path, default=None, help="Optional RGB with black background")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-type", default="vit_b", choices=("vit_h", "vit_l", "vit_b"))
    parser.add_argument(
        "--mode",
        default="center_subject",
        choices=("center_subject", "center_point", "auto_masks_center_bias", "auto_masks_largest_area"),
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--no-fp16",
        dest="fp16",
        action="store_false",
        help="Disable torch.cuda.amp fp16 on CUDA",
    )
    parser.set_defaults(fp16=True)
    args = parser.parse_args()

    try:
        import torch
        from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
    except ImportError as e:
        print("Install: pip install segment-anything torch torchvision", file=sys.stderr)
        raise SystemExit(1) from e

    if not args.checkpoint.is_file():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type != "cuda":
        print("Warning: CUDA not available; running on CPU.", file=sys.stderr)

    sam = sam_model_registry[args.model_type](checkpoint=str(args.checkpoint))
    sam.to(device=device)
    sam.eval()

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = sorted(p for p in args.input_dir.iterdir() if p.suffix.lower() in exts)
    if len(paths) < 1:
        print(f"No images in {args.input_dir}", file=sys.stderr)
        return 1

    args.mask_dir.mkdir(parents=True, exist_ok=True)
    if args.masked_dir:
        args.masked_dir.mkdir(parents=True, exist_ok=True)

    predictor = SamPredictor(sam)
    mask_gen = SamAutomaticMaskGenerator(sam)

    use_amp = args.fp16 and device.type == "cuda"

    def _amp():
        return torch.cuda.amp.autocast(dtype=torch.float16) if use_amp else contextlib.nullcontext()

    def predict_one(rgb: np.ndarray, h: int, w: int) -> np.ndarray | None:
        if args.mode in ("center_point", "center_subject"):
            with torch.inference_mode():
                with _amp():
                    predictor.set_image(rgb)
                    cx, cy = w // 2, h // 2
                    if args.mode == "center_point":
                        coords = np.array([[float(cx), float(cy)]], dtype=np.float32)
                        labels = np.array([1], dtype=np.int32)
                    else:
                        coords = np.array(
                            [
                                [float(cx), float(cy)],
                                [0.0, 0.0],
                                [float(w - 1), 0.0],
                                [0.0, float(h - 1)],
                                [float(w - 1), float(h - 1)],
                            ],
                            dtype=np.float32,
                        )
                        labels = np.array([1, 0, 0, 0, 0], dtype=np.int32)
                    masks, scores, _ = predictor.predict(
                        point_coords=coords,
                        point_labels=labels,
                        multimask_output=True,
                    )
            if masks is None or len(masks) == 0:
                return None
            raw = masks[int(np.argmax(scores))].astype(np.uint8) * 255
            return refine_binary_mask_to_center_subject(raw)

        if args.mode == "auto_masks_largest_area":
            with torch.inference_mode():
                with _amp():
                    masks = mask_gen.generate(rgb)
            if not masks:
                return None
            best = max(masks, key=lambda m: int(m.get("area", 0)))
            return (best["segmentation"].astype(np.uint8) * 255)

        # auto_masks_center_bias
        cx_img, cy_img = w / 2.0, h / 2.0
        max_dist = float(np.hypot(cx_img, cy_img)) + 1e-6

        def score(m: dict) -> float:
            seg = m["segmentation"]
            area = float(m.get("area", float(seg.sum())))
            if area < 64.0:
                return 0.0
            ys, xs = np.where(seg)
            if len(xs) == 0:
                return 0.0
            mx, my = float(xs.mean()), float(ys.mean())
            dist = float(np.hypot(mx - cx_img, my - cy_img))
            norm_dist = dist / max_dist
            center_w = float(np.exp(-3.0 * norm_dist**2))
            return float(np.log1p(area)) * center_w

        with torch.inference_mode():
            with _amp():
                masks = mask_gen.generate(rgb)
        if not masks:
            return None
        best = max(masks, key=score)
        raw = best["segmentation"].astype(np.uint8) * 255
        return refine_binary_mask_to_center_subject(raw)

    for i, p in enumerate(paths):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"Skip unreadable: {p}", file=sys.stderr)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        mask = predict_one(rgb, h, w)
        if mask is None:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (w // 2, h // 2), (max(1, w // 3), max(1, h // 3)), 0, 0, 360, 255, -1)

        stem = p.stem
        mask_path = args.mask_dir / f"{stem}_mask.png"
        cv2.imwrite(str(mask_path), mask)
        if args.masked_dir:
            binary = (mask > 0).astype(np.uint8)
            masked = bgr * binary[:, :, None]
            cv2.imwrite(str(args.masked_dir / f"{stem}.png"), masked)
        print(f"[{i + 1}/{len(paths)}] {p.name} -> {mask_path.name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
