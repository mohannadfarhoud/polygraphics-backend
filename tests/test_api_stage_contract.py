import json
import unittest
from pathlib import Path


class ApiStageContractTests(unittest.TestCase):
    def test_surface_region_stages_are_published(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        stages_doc = json.loads((repo_root / "ui" / "job-stages-progress.json").read_text(encoding="utf-8"))
        stage_ids = {entry.get("id") for entry in stages_doc.get("stages", [])}
        self.assertIn("phase_surface_region_extract", stage_ids)
        self.assertIn("phase_surface_region_projection", stage_ids)
        self.assertIn("phase_surface_region_blend", stage_ids)


if __name__ == "__main__":
    unittest.main()
