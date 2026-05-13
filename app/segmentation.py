from __future__ import annotations

import contextlib
from pathlib import Path

import cv2
import numpy as np

from .runtime_settings import RuntimeSettings


class SamSegmenter:
    """SAM-backed masking; optional ellipse demo when allow_placeholder_pipeline is True."""

    def __init__(self, settings: RuntimeSettings | None = None) -> None:
        self.settings = settings
        self._sam_model = None
        self._mask_generator = None
        self._predictor = None

    def predict_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.settings is not None and self.settings.allow_placeholder_pipeline:
            return self._fallback_center_mask(image_bgr)

        ckpt = self.settings.sam_checkpoint_path if self.settings else None
        if not ckpt or not Path(ckpt).is_file():
            raise RuntimeError("SAM checkpoint missing or invalid (sam_checkpoint_path).")

        try:
            import torch
            from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
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

        mode = self.settings.sam_segmentation_mode if self.settings else "center_point"
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        sam = self._ensure_sam(model_type, ckpt, device)

        use_amp = (
            device.type == "cuda"
            and self.settings is not None
            and bool(getattr(self.settings, "sam_use_fp16", True))
        )
        if use_amp:
            amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16)
        else:
            amp_ctx = contextlib.nullcontext()

        with amp_ctx:
            if mode == "center_point":
                mask = self._predict_center_point(sam, image_rgb, h, w)
            elif mode == "auto_masks_largest_area":
                mask = self._predict_auto_largest(sam, image_rgb)
            else:
                mask = self._predict_auto_center_bias(sam, image_rgb, h, w)

        if mask is None or mask.size == 0:
            return self._fallback_center_mask(image_bgr)
        return mask

    def _ensure_sam(self, model_type: str, ckpt: str, device):
        from segment_anything import sam_model_registry

        if self._sam_model is None:
            sam = sam_model_registry[model_type](checkpoint=ckpt)
            sam.to(device=device)
            sam.eval()
            self._sam_model = sam
        return self._sam_model

    def _predict_center_point(self, sam, image_rgb: np.ndarray, h: int, w: int) -> np.ndarray:
        """Prompt SAM with a positive point at the image center — best for a subject in the middle."""
        from segment_anything import SamPredictor

        if self._predictor is None:
            self._predictor = SamPredictor(sam)

        self._predictor.set_image(image_rgb)
        cx, cy = w // 2, h // 2
        point_coords = np.array([[float(cx), float(cy)]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)
        masks, scores, _logits = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        if masks is None or len(masks) == 0:
            return None
        best_idx = int(np.argmax(scores))
        seg = masks[best_idx]
        return (seg.astype(np.uint8) * 255)

    def _predict_auto_largest(self, sam, image_rgb: np.ndarray) -> np.ndarray:
        """Original behavior: largest automatic mask by pixel area."""
        from segment_anything import SamAutomaticMaskGenerator

        if self._mask_generator is None:
            self._mask_generator = SamAutomaticMaskGenerator(sam)

        masks = self._mask_generator.generate(image_rgb)
        if not masks:
            return None
        best = max(masks, key=lambda m: int(m.get("area", 0)))
        seg = best["segmentation"]
        return (seg.astype(np.uint8) * 255)

    def _predict_auto_center_bias(self, sam, image_rgb: np.ndarray, h: int, w: int) -> np.ndarray:
        """Score each auto-mask by log(area) × Gaussian falloff from image center."""
        from segment_anything import SamAutomaticMaskGenerator

        if self._mask_generator is None:
            self._mask_generator = SamAutomaticMaskGenerator(sam)

        masks = self._mask_generator.generate(image_rgb)
        if not masks:
            return None

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

        best = max(masks, key=score)
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

    def segment_file(
        self,
        image_path: Path,
        output_path: Path,
        *,
        mask_output_path: Path | None = None,
    ) -> Path:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image: {image_path}")
        mask = self.predict_mask(image)
        masked = self.apply_black_background(image, mask)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), masked)
        if mask_output_path is not None:
            mask_output_path.parent.mkdir(parents=True, exist_ok=True)
            binary_mask = ((mask > 0).astype(np.uint8)) * 255
            cv2.imwrite(str(mask_output_path), binary_mask)
        return output_path
