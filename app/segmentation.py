from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .runtime_settings import RuntimeSettings


class SamSegmenter:
    """SAM-backed masking; optional ellipse demo when allow_placeholder_pipeline is True."""

    def __init__(self, settings: RuntimeSettings | None = None) -> None:
        self.settings = settings
        self._mask_generator = None

    def predict_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.settings is not None and self.settings.allow_placeholder_pipeline:
            return self._fallback_center_mask(image_bgr)

        ckpt = self.settings.sam_checkpoint_path if self.settings else None
        if not ckpt or not Path(ckpt).is_file():
            raise RuntimeError("SAM checkpoint missing or invalid (sam_checkpoint_path).")

        try:
            import torch
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        except ImportError as exc:
            raise RuntimeError(
                "segment_anything is not installed. Install with: pip install segment-anything torch torchvision. "
                f"Original: {exc}"
            ) from exc

        model_type = self.settings.sam_model_type if self.settings else "vit_h"
        if model_type not in sam_model_registry:
            raise RuntimeError(f"Unknown sam_model_type {model_type!r}; use vit_h, vit_l, or vit_b.")

        device_str = self.settings.device if self.settings else "auto"
        if device_str == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_str)

        if self._mask_generator is None:
            sam = sam_model_registry[model_type](checkpoint=ckpt)
            sam.to(device=device)
            sam.eval()
            self._mask_generator = SamAutomaticMaskGenerator(sam)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        masks = self._mask_generator.generate(image_rgb)
        if not masks:
            return self._fallback_center_mask(image_bgr)

        best = max(masks, key=lambda m: int(m.get("area", 0)))
        seg = best["segmentation"]
        return (seg.astype(np.uint8) * 255)

    @staticmethod
    def _fallback_center_mask(image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cx, cy = w // 2, h // 2
        rx, ry = max(1, int(w * 0.3)), max(1, int(h * 0.3))
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
        return mask

    def apply_black_background(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask.ndim != 2:
            raise ValueError("Mask must be a single-channel binary image")
        binary_mask = (mask > 0).astype(np.uint8)
        return image_bgr * binary_mask[:, :, None]

    def segment_file(self, image_path: Path, output_path: Path) -> Path:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        mask = self.predict_mask(image)
        masked = self.apply_black_background(image, mask)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), masked)
        return output_path
