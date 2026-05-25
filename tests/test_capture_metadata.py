import unittest

from app.capture_metadata import (
    capture_metadata_response_fields,
    parse_capture_metadata_form,
    validate_capture_metadata_payload,
)


class CaptureMetadataTests(unittest.TestCase):
    def test_validate_derives_summary_when_missing(self) -> None:
        payload = {
            "schema_version": "1.0",
            "frames": [
                {"file": "a.jpg", "yaw_deg": -10, "quality_score": 0.8, "accepted": True},
                {"file": "b.jpg", "yaw_deg": 50, "quality_score": 0.6, "accepted": False},
                {"file": "c.jpg", "yaw_deg": 120, "quality_score": 0.7, "accepted": True},
            ],
        }
        normalized = validate_capture_metadata_payload(payload)
        summary = normalized.get("summary", {})
        self.assertEqual(summary.get("total_frames"), 3)
        self.assertEqual(summary.get("accepted_frames"), 2)
        self.assertEqual(summary.get("rejected_frames"), 1)
        self.assertAlmostEqual(float(summary.get("avg_quality_score")), 0.7, places=4)
        self.assertAlmostEqual(float(summary.get("orbit_coverage_deg")), 130.0, places=4)

    def test_parse_capture_metadata_form_requires_json_object(self) -> None:
        with self.assertRaises(ValueError):
            parse_capture_metadata_form('["not", "an", "object"]')

    def test_response_fields_extract_summary(self) -> None:
        payload = {
            "schema_version": "2.1",
            "summary": {
                "total_frames": 24,
                "accepted_frames": 20,
                "avg_quality_score": 0.81,
                "orbit_coverage_deg": 278.5,
            },
        }
        version, total, accepted, avg_quality, coverage = capture_metadata_response_fields(payload)
        self.assertEqual(version, "2.1")
        self.assertEqual(total, 24)
        self.assertEqual(accepted, 20)
        self.assertAlmostEqual(float(avg_quality), 0.81, places=4)
        self.assertAlmostEqual(float(coverage), 278.5, places=4)


if __name__ == "__main__":
    unittest.main()
