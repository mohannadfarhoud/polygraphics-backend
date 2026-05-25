import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.surface_region_texture import build_dominant_surface_texture_images


class SurfaceRegionTextureTests(unittest.TestCase):
    def test_extracts_region_and_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "view.png"
            img = np.zeros((128, 128, 3), dtype=np.uint8)
            img[:, 10:110, :] = (170, 170, 170)
            img[:, 100:128, :] = (30, 30, 30)
            cv2.imwrite(str(src), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            result = build_dominant_surface_texture_images(
                job_id="job_t",
                image_paths=[src],
                cache_root=root / "cache",
                smooth_percentile=85.0,
                min_area_ratio=0.05,
                expand_px=4,
            )
            self.assertIn(str(src.resolve()), result.remap)
            self.assertEqual(len(result.entries), 1)
            entry = result.entries[0]
            self.assertGreaterEqual(entry.confidence, 0.0)
            self.assertLessEqual(entry.confidence, 1.0)
            self.assertGreater(entry.area_ratio, 0.05)
            self.assertIn(entry.label_hint, {"left", "right", "front", "top"})


if __name__ == "__main__":
    unittest.main()
