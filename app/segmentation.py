from __future__ import annotations

import contextlib
from pathlib import Path

import cv2
import numpy as np

from .runtime_settings import RuntimeSettings


def refine_binary_mask_to_center_subject(mask_u8: np.ndarray) -> np.ndarray:
    """Keep the connected foreground component that contains the image centre (or nearest large blob)."""
    h, w = mask_u8.shape[:2]
    binary = ((mask_u8 > 0).astype(np.uint8) * 255).astype(np.uint8)
    cx, cy = w // 2, h // 2
    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return binary

    lid = int(labels[cy, cx])
    if lid > 0 and stats[lid, cv2.CC_STAT_AREA] >= 32:
        return ((labels == lid).astype(np.uint8) * 255)

    best_li = -1
    best_score = -1.0
    for li in range(1, n):
        area = int(stats[li, cv2.CC_STAT_AREA])
        if area < 32:
            continue
        m = labels == li
        ys, xs = np.where(m)
        dist2 = float((xs.mean() - cx) ** 2 + (ys.mean() - cy) ** 2)
        score = float(area) / (1.0 + dist2 * 1e-4)
        if score > best_score:
            best_score = score
            best_li = li
    if best_li < 0:
        return binary
    return ((labels == best_li).astype(np.uint8) * 255)


class SamSegmenter:
    """SAM-backed masking for **isolating a centred foreground object** from scene + tabletop.

    Default ``sam_segmentation_mode=center_subject_table`` adds bottom-edge background prompts so the flat
    surface under the subject is excluded more often than with corners-only ``center_subject``.
    Ellipse fallback only when ``allow_placeholder_pipeline`` is True.
    """

    def __init__(self, settings: RuntimeSettings | None = None) -> None:
        self.settings = settings
        self._sam_model = None
        self._mask_generator = None
        self._predictor = None
        self._rembg_session = None
        self._rembg_model_name = None

    def predict_mask(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.settings is not None and self.settings.allow_placeholder_pipeline:
            return self._fallback_center_mask(image_bgr)

        backend = (getattr(self.settings, "isolation_backend", "sam") if self.settings else "sam").strip().lower()
        if backend == "rembg":
            mask = self._predict_rembg_mask(image_bgr)
            if mask is None or mask.size == 0:
                return self._fallback_center_mask(image_bgr)
            return mask

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

        model_type = self.settings.sam_model_type if self.settings else "vit_b"
        if model_type not in sam_model_registry:
            raise RuntimeError(f"Unknown sam_model_type {model_type!r}; use vit_h, vit_l, or vit_b.")

        device_str = self.settings.device if self.settings else "auto"
        if device_str == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_str)

        mode = self.settings.sam_segmentation_mode if self.settings else "center_subject_table"
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
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
            if mode == "center_subject_table":
                mask = self._predict_center_subject_table(sam, image_rgb)
            elif mode == "center_subject":
                mask = self._predict_center_subject(sam, image_rgb)
            elif mode == "center_point":
                mask = self._predict_center_point(sam, image_rgb)
            elif mode == "auto_masks_largest_area":
                mask = self._predict_auto_largest(sam, image_rgb)
            else:
                mask = self._predict_auto_center_bias(sam, image_rgb)

        if mask is None or mask.size == 0:
            return self._fallback_center_mask(image_bgr)
        return mask

    def _predict_rembg_mask(self, image_bgr: np.ndarray) -> np.ndarray | None:
        """Isolate foreground via rembg alpha matte, then keep the centre subject component."""
        try:
            from rembg import new_session, remove
        except ImportError as exc:
            raise RuntimeError(
                "rembg is not installed. Install with: pip install rembg onnxruntime pillow"
            ) from exc

        model_name = (
            str(getattr(self.settings, "rembg_model_name", "isnet-general-use")).strip()
            if self.settings
            else "isnet-general-use"
        )
        if not model_name:
            model_name = "isnet-general-use"
        if self._rembg_session is None or self._rembg_model_name != model_name:
            self._rembg_session = new_session(model_name)
            self._rembg_model_name = model_name

        ok, enc = cv2.imencode(".png", image_bgr)
        if not ok:
            return None
        out_bytes = remove(enc.tobytes(), session=self._rembg_session)
        rgba = cv2.imdecode(np.frombuffer(out_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] < 4:
            return None
        alpha = rgba[:, :, 3]
        thr = int(getattr(self.settings, "rembg_alpha_threshold", 16)) if self.settings else 16
        thr = max(0, min(255, thr))
        raw = ((alpha >= thr).astype(np.uint8) * 255).astype(np.uint8)
        return refine_binary_mask_to_center_subject(raw)

    def release_gpu_memory(self) -> None:
        """Drop Segment Anything tensors so MapAnything / GS fit on ~8 GB GPUs (worker single-process)."""
        self._sam_model = None
        self._predictor = None
        self._mask_generator = None
        try:
            from .cuda_memory import purge_torch_cuda

            purge_torch_cuda()
        except Exception:
            pass

    def _ensure_sam(self, model_type: str, ckpt: str, device):
        from segment_anything import sam_model_registry

        if self._sam_model is None:
            sam = sam_model_registry[model_type](checkpoint=ckpt)
            sam.to(device=device)
            sam.eval()
            self._sam_model = sam
        return self._sam_model

    @staticmethod
    def _mask_area_ratio(mask: np.ndarray) -> float:
        if mask is None or mask.size == 0:
            return 0.0
        return float(np.mean(mask > 0))

    def _prompt_area_bounds(self) -> tuple[float, float]:
        if self.settings is None:
            return 0.0005, 0.45
        mn = float(getattr(self.settings, "sam_prompt_min_mask_area_ratio", 0.0005))
        mx = float(getattr(self.settings, "sam_prompt_max_mask_area_ratio", 0.45))
        mn = max(0.0, min(0.2, mn))
        mx = max(0.05, min(0.98, mx))
        if mn >= mx:
            mn = max(0.0, mx * 0.1)
        return mn, mx

    def _choose_prompt_mask(self, masks: np.ndarray, scores: np.ndarray | None, h: int, w: int) -> np.ndarray | None:
        """Select the best candidate mask from prompt-based SAM outputs.

        SAM can return a high-confidence but background-heavy region on textured scenes.
        We score candidates with center preference + area sanity (small centered subject).
        """
        if masks is None or len(masks) == 0:
            return None
        cx, cy = w // 2, h // 2
        min_ratio, max_ratio = self._prompt_area_bounds()
        require_center = bool(getattr(self.settings, "sam_prompt_require_center_hit", True)) if self.settings else True
        center_candidates: list[tuple[np.ndarray, float, float, int]] = []
        all_candidates: list[tuple[np.ndarray, float, float, int]] = []
        for i, seg in enumerate(masks):
            raw = seg.astype(np.uint8) * 255
            refined = refine_binary_mask_to_center_subject(raw)
            area = self._mask_area_ratio(refined)
            if area <= 0.0:
                continue
            sam_score = float(scores[i]) if scores is not None and i < len(scores) else 0.0
            center_hit = int(refined[cy, cx] > 0)
            all_candidates.append((refined, area, sam_score, center_hit))
            if center_hit:
                center_candidates.append((refined, area, sam_score, center_hit))

        candidates = center_candidates if center_candidates else all_candidates
        if require_center and not center_candidates:
            return None
        best_mask = None
        best_score = float("-inf")
        for refined, area, sam_score, center_hit in candidates:
            over = max(0.0, area - max_ratio)
            under = max(0.0, min_ratio - area)
            # Favor center-hit strongly, discourage massive-background or tiny speck masks.
            objective = sam_score + (2.5 * float(center_hit)) - (8.0 * over) - (3.0 * under)
            if objective > best_score:
                best_score = objective
                best_mask = refined
        return best_mask

    def _recover_bad_prompt_mask(self, sam, image_rgb: np.ndarray, prompt_mask: np.ndarray) -> np.ndarray:
        """If prompt mask looks implausible, recover with auto center-biased SAM."""
        recover = bool(getattr(self.settings, "sam_recover_with_auto_if_prompt_bad", True)) if self.settings else True
        if prompt_mask is None:
            if not recover:
                return None
            return self._predict_auto_center_bias(sam, image_rgb)
        if not recover:
            return prompt_mask
        area = self._mask_area_ratio(prompt_mask)
        min_ratio, max_ratio = self._prompt_area_bounds()
        if min_ratio <= area <= max_ratio:
            return prompt_mask
        alt = self._predict_auto_center_bias(sam, image_rgb)
        if alt is None or alt.size == 0:
            return prompt_mask
        alt_area = self._mask_area_ratio(alt)
        if alt_area <= 0.0:
            return prompt_mask
        # Prefer recovered mask when it is in-bounds, or clearly tighter than the prompt result.
        if (min_ratio <= alt_area <= max_ratio) or (alt_area < area * 0.7):
            return alt
        return prompt_mask

    def _predict_center_point(self, sam, image_rgb: np.ndarray) -> np.ndarray:
        """Prompt SAM with a positive point at the image center — best for a subject in the middle."""
        from segment_anything import SamPredictor

        if self._predictor is None:
            self._predictor = SamPredictor(sam)

        self._predictor.set_image(image_rgb)
        h, w = image_rgb.shape[:2]
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
        h, w = image_rgb.shape[:2]
        chosen = self._choose_prompt_mask(masks, scores, h, w)
        return self._recover_bad_prompt_mask(sam, image_rgb, chosen)

    def _predict_center_subject(self, sam, image_rgb: np.ndarray) -> np.ndarray:
        """Centre foreground prompt + corner background prompts, then keep the FG component touching the centre.

        Stronger background removal than ``center_point`` alone when the backdrop is homogeneous.
        """
        from segment_anything import SamPredictor

        if self._predictor is None:
            self._predictor = SamPredictor(sam)

        self._predictor.set_image(image_rgb)
        h, w = image_rgb.shape[:2]
        cx, cy = w // 2, h // 2
        point_coords = np.array(
            [
                [float(cx), float(cy)],
                [0.0, 0.0],
                [float(w - 1), 0.0],
                [0.0, float(h - 1)],
                [float(w - 1), float(h - 1)],
            ],
            dtype=np.float32,
        )
        point_labels = np.array([1, 0, 0, 0, 0], dtype=np.int32)
        masks, scores, _logits = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        if masks is None or len(masks) == 0:
            return None
        h, w = image_rgb.shape[:2]
        chosen = self._choose_prompt_mask(masks, scores, h, w)
        return self._recover_bad_prompt_mask(sam, image_rgb, chosen)

    def _predict_center_subject_table(self, sam, image_rgb: np.ndarray) -> np.ndarray:
        """Centre foreground + corner background + negatives along bottom edge (table plane).

        SAM often merges tabletop with the object when only corners are marked background; discouraging
        the bottom strip biases the mask toward the lifted/rigid subject above the surface.
        """
        from segment_anything import SamPredictor

        if self._predictor is None:
            self._predictor = SamPredictor(sam)

        self._predictor.set_image(image_rgb)
        h, w = image_rgb.shape[:2]
        cx, cy = w // 2, h // 2
        coords = [
            [float(cx), float(cy)],
            [0.0, 0.0],
            [float(w - 1), 0.0],
            [0.0, float(h - 1)],
            [float(w - 1), float(h - 1)],
        ]
        labels = [1, 0, 0, 0, 0]

        n_edge = int(getattr(self.settings, "sam_table_edge_negative_points", 11)) if self.settings else 11
        n_edge = max(0, min(24, n_edge))

        if n_edge > 0:
            margin = float(max(2.0, min(w, h) * 0.035))
            margin = min(margin, max(1.0, (w - 2) / 2.01))
            inset_y = max(1, min(h // 60, 12))
            yb = float(h - 1 - inset_y)
            xs = np.linspace(margin, float(w - 1) - margin, num=n_edge, dtype=np.float64)
            for xv in xs:
                coords.append([float(xv), yb])
                labels.append(0)

        point_coords = np.array(coords, dtype=np.float32)
        point_labels = np.array(labels, dtype=np.int32)
        masks, scores, _logits = self._predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        if masks is None or len(masks) == 0:
            return None
        chosen = self._choose_prompt_mask(masks, scores, h, w)
        return self._recover_bad_prompt_mask(sam, image_rgb, chosen)

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

    def _predict_auto_center_bias(self, sam, image_rgb: np.ndarray) -> np.ndarray:
        """Score each auto-mask by log(area) × Gaussian falloff from image center."""
        from segment_anything import SamAutomaticMaskGenerator

        if self._mask_generator is None:
            self._mask_generator = SamAutomaticMaskGenerator(sam)

        masks = self._mask_generator.generate(image_rgb)
        if not masks:
            return None

        h, w = image_rgb.shape[:2]
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
        raw = (seg.astype(np.uint8) * 255)
        return refine_binary_mask_to_center_subject(raw)

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

    @staticmethod
    def _recenter_masked_subject(
        masked_bgr: np.ndarray,
        mask_u8: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Translate isolated foreground so its centroid is at the frame center."""
        if mask_u8.ndim != 2:
            return masked_bgr, mask_u8
        m = (mask_u8 > 0).astype(np.uint8)
        if int(m.sum()) == 0:
            return masked_bgr, mask_u8
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            return masked_bgr, mask_u8
        h, w = m.shape[:2]
        obj_cx = float(xs.mean())
        obj_cy = float(ys.mean())
        tgt_cx = (w - 1) / 2.0
        tgt_cy = (h - 1) / 2.0
        dx = int(round(tgt_cx - obj_cx))
        dy = int(round(tgt_cy - obj_cy))
        if dx == 0 and dy == 0:
            return masked_bgr, mask_u8
        M = np.float32([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]])
        recentered_masked = cv2.warpAffine(
            masked_bgr,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        recentered_mask = cv2.warpAffine(
            ((mask_u8 > 0).astype(np.uint8) * 255),
            M,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return recentered_masked, recentered_mask

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
        if bool(getattr(self.settings, "recenter_isolated_subject", True)):
            masked, mask = self._recenter_masked_subject(masked, mask)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), masked)
        if mask_output_path is not None:
            mask_output_path.parent.mkdir(parents=True, exist_ok=True)
            binary_mask = ((mask > 0).astype(np.uint8)) * 255
            cv2.imwrite(str(mask_output_path), binary_mask)
        return output_path
