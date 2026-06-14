import gc
import importlib
import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.photo_compose_util import placement_pixels


class PhotoComposeValidationTests(unittest.TestCase):
    def test_placement_pixels(self) -> None:
        px, py = placement_pixels(0.45, 0.55, 1080, 1920)
        self.assertEqual(px, 486)
        self.assertEqual(py, 1056)


class PhotoComposeApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self._root, ignore_errors=True))
        for sub in ("data", "output", "uploads", "config"):
            (self._root / sub).mkdir(parents=True, exist_ok=True)
        (self._root / "config" / "runtime_settings.json").write_text(
            json.dumps({"output_dir_name": "output"}),
            encoding="utf-8",
        )
        job_id = "earring-job"
        (self._root / "output" / f"{job_id}.glb").write_bytes(b"glb")
        job_dir = self._root / "uploads" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        # Minimal JPEG placeholder (no Pillow required in tests)
        (job_dir / "input_0.jpg").write_bytes(
            bytes.fromhex(
                "ffd8ffe000104a46494600010100000100010000ffdb004300080606"
                "070605080707070909080a0c140d0c0b0b0c1912130f141d1a1f1e"
                "1d1a1c1c20242e2720222c231c1c2837292c30313434341f27393d"
                "38323c2e333432ffdb0043010909090c0b0c180d0d1832211c"
                "2132323232323232323232323232323232323232323232323232"
                "323232323232ffc00011080001000103011100021100031100ffc4"
                "0014000100000000000000000000000000000008ffc400141001"
                "00000000000000000000000000000000ffda0008010100003f00"
                "d2cfd0ffc4"
            )
        )

        self._env = patch.dict(
            "os.environ",
            {
                "APP_ROOT_DIR": str(self._root),
                "APP_DATABASE_PATH": str(self._root / "data" / "test.sqlite"),
                "APP_TRY_ON_BIND_OWNER": "0",
                "PHOTO_COMPOSE_DEV_MOCK": "1",
                "GEMINI_API_KEY": "",
            },
            clear=False,
        )
        self._env.start()
        import app.main as main_mod

        importlib.reload(main_mod)
        self.client = TestClient(main_mod.app)
        self.job_id = job_id

    def tearDown(self) -> None:
        self._env.stop()
        gc.collect()

    def _face_jpeg(self) -> bytes:
        return bytes.fromhex(
            "ffd8ffe000104a46494600010100000100010000ffdb004300080606"
            "070605080707070909080a0c140d0c0b0b0c1912130f141d1a1f1e"
            "1d1a1c1c20242e2720222c231c1c2837292c30313434341f27393d"
            "38323c2e333432ffdb0043010909090c0b0c180d0d1832211c"
            "2132323232323232323232323232323232323232323232323232"
            "323232323232ffc00011080001000103011100021100031100ffc4"
            "0014000100000000000000000000000000000008ffc400141001"
            "00000000000000000000000000000000ffda0008010100003f00"
            "d2cfd0ffc4"
        )

    def test_post_and_poll_compose(self) -> None:
        files = {"face_image": ("face.jpg", self._face_jpeg(), "image/jpeg")}
        data = {
            "job_id": self.job_id,
            "placement_x": "0.45",
            "placement_y": "0.55",
            "image_width": "1080",
            "image_height": "1920",
            "placement_side": "auto",
        }
        post = self.client.post("/try-on/photo-compose", files=files, data=data)
        self.assertEqual(post.status_code, 202, post.text)
        body = post.json()
        self.assertIn(body["status"], ("queued", "processing", "completed"))
        compose_id = body["compose_id"]

        deadline = time.time() + 15.0
        final = None
        while time.time() < deadline:
            got = self.client.get(f"/try-on/photo-compose/{compose_id}")
            self.assertEqual(got.status_code, 200)
            final = got.json()
            if final["status"] in ("completed", "failed"):
                break
            time.sleep(0.25)

        assert final is not None
        self.assertEqual(final["status"], "completed")
        self.assertTrue(final["result_url"])
        self.assertIn("photo-compose", final["result_url"])

    def test_missing_image_url_422(self) -> None:
        (self._root / "output" / "no-thumb.glb").write_bytes(b"glb")
        files = {"face_image": ("face.jpg", self._face_jpeg(), "image/jpeg")}
        data = {
            "job_id": "no-thumb",
            "placement_x": "0.5",
            "placement_y": "0.5",
            "image_width": "100",
            "image_height": "100",
        }
        r = self.client.post("/try-on/photo-compose", files=files, data=data)
        self.assertEqual(r.status_code, 422)

    def test_invalid_placement_400(self) -> None:
        files = {"face_image": ("face.jpg", self._face_jpeg(), "image/jpeg")}
        data = {
            "job_id": self.job_id,
            "placement_x": "1.5",
            "placement_y": "0.5",
            "image_width": "1080",
            "image_height": "1920",
        }
        r = self.client.post("/try-on/photo-compose", files=files, data=data)
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
