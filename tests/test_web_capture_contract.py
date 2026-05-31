import unittest

from app.runtime_settings import RuntimeSettings
from app.web_capture_contract import build_web_capture_contract


class WebCaptureContractTests(unittest.TestCase):
    def test_contract_reflects_runtime_thresholds(self) -> None:
        settings = RuntimeSettings(
            reconstruction_backend="ai_prior",
            ai_prior_provider="triposr_local",
            ai_prior_force_prior_only=True,
            capture_blur_min=44.0,
            capture_brightness_min=30.0,
            capture_brightness_max=205.0,
            capture_min_frame_delta=0.017,
            capture_duplicate_similarity=0.993,
            capture_diversity_min_distance=0.052,
            capture_reject_policy="hard",
            capture_min_kept_images=10,
            capture_max_selected_images=24,
            depth_consistency_max_relative=0.25,
            max_images=80,
        )
        payload = build_web_capture_contract(settings)
        self.assertEqual(payload["pipeline"]["reconstruction_backend"], "ai_prior")
        self.assertEqual(payload["pipeline"]["ai_prior_provider"], "triposr_local")
        self.assertEqual(payload["pipeline"]["minimum_input_images"], 1)

        checks = payload["live_frame_checks"]
        self.assertAlmostEqual(float(checks["blur"]["min"]), 44.0, places=4)
        self.assertAlmostEqual(float(checks["brightness"]["min"]), 30.0, places=4)
        self.assertAlmostEqual(float(checks["brightness"]["max"]), 205.0, places=4)
        self.assertAlmostEqual(float(checks["frame_delta"]["min"]), 0.017, places=4)
        self.assertAlmostEqual(float(checks["duplicate_similarity"]["max"]), 0.993, places=4)
        self.assertAlmostEqual(float(checks["diversity_distance"]["min"]), 0.052, places=4)

    def test_contract_recommended_counts_are_reasonable(self) -> None:
        settings = RuntimeSettings(
            reconstruction_backend="mapanything",
            max_images=30,
        )
        payload = build_web_capture_contract(settings)
        count = payload["capture_targets"]["image_count"]
        self.assertEqual(int(count["hard_minimum"]), 2)
        self.assertGreaterEqual(int(count["warn_below"]), 12)
        self.assertLessEqual(int(count["target_min"]), 30)
        self.assertLessEqual(int(count["target_max"]), 30)
        self.assertGreaterEqual(int(count["target_max"]), int(count["target_min"]))


if __name__ == "__main__":
    unittest.main()

