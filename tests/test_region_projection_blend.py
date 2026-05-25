import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from app.color_baking import CameraView, bake_vertex_colors_from_views


class RegionProjectionBlendTests(unittest.TestCase):
    def test_projection_and_blending_produce_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            img_path = root / "view.png"
            rgb = np.zeros((64, 64, 3), dtype=np.uint8)
            rgb[:, :, :] = (220, 20, 20)
            cv2.imwrite(str(img_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(
                np.asarray(
                    [
                        [-0.2, -0.2, 1.0],
                        [0.2, -0.2, 1.0],
                        [0.0, 0.2, 1.0],
                    ],
                    dtype=np.float64,
                )
            )
            mesh.triangles = o3d.utility.Vector3iVector(np.asarray([[0, 1, 2]], dtype=np.int32))

            K = np.asarray([[60.0, 0.0, 32.0], [0.0, 60.0, 32.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            w2c = np.eye(4, dtype=np.float64)
            view = CameraView(
                image_path=img_path,
                image_size=(64, 64),
                K=K,
                w2c=w2c,
                surface_region_confidence=0.9,
                surface_region_label="front",
            )

            ok, diag = bake_vertex_colors_from_views(
                mesh,
                [view],
                blend_weight=0.65,
                min_confidence=0.3,
                seam_smoothing=0.55,
            )
            self.assertTrue(ok)
            self.assertGreater(diag.dominant_surface_regions_detected, 0)
            self.assertIn("mesh_area_ratio", diag.region_projection_coverage)


if __name__ == "__main__":
    unittest.main()
