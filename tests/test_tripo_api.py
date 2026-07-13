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


class TripoApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self._root, ignore_errors=True))
        for sub in ("data", "output", "uploads", "config"):
            (self._root / sub).mkdir(parents=True, exist_ok=True)
        (self._root / "config" / "runtime_settings.json").write_text(
            json.dumps({"output_dir_name": "output"}),
            encoding="utf-8",
        )

        self._env = patch.dict(
            "os.environ",
            {
                "APP_ROOT_DIR": str(self._root),
                "APP_DATABASE_PATH": str(self._root / "data" / "test.sqlite"),
                "APP_REMOTE_WORKERS": "0",
                "TRIPO_DEV_MOCK": "1",
                "TRIPO_API_KEY": "",
            },
            clear=False,
        )
        self._env.start()
        import app.main as main_mod

        importlib.reload(main_mod)
        self.client = TestClient(main_mod.app)

    def tearDown(self) -> None:
        self._env.stop()
        gc.collect()

    def _jpeg_bytes(self) -> bytes:
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

    def test_image_to_model_flow(self) -> None:
        files = {"image": ("product.jpg", self._jpeg_bytes(), "image/jpeg")}
        post = self.client.post("/tripo/image-to-model", files=files)
        self.assertEqual(post.status_code, 202, post.text)
        body = post.json()
        tripo_job_id = body["tripo_job_id"]
        self.assertEqual(body["status"], "queued")

        final = None
        for _ in range(40):
            got = self.client.get(f"/tripo/image-to-model/{tripo_job_id}")
            self.assertEqual(got.status_code, 200, got.text)
            final = got.json()
            if final["status"] in ("completed", "failed"):
                break
            time.sleep(0.05)

        self.assertIsNotNone(final)
        assert final is not None
        self.assertEqual(final["status"], "completed")
        self.assertIn("download_url", final)
        self.assertTrue(final["download_url"].endswith("/download"))

        dl = self.client.get(final["download_url"])
        self.assertEqual(dl.status_code, 200, dl.text)
        self.assertTrue(dl.content.startswith(b"glTF"))

    def test_works_without_auth(self) -> None:
        files = {"image": ("product.jpg", self._jpeg_bytes(), "image/jpeg")}
        r = self.client.post("/tripo/image-to-model", files=files)
        self.assertEqual(r.status_code, 202, r.text)

    def test_auto_mock_without_api_key(self) -> None:
        import app.tripo_core as tripo_core

        self.assertTrue(tripo_core.tripo_configured())
        self.assertEqual(tripo_core.resolve_tripo_backend(), "dev_mock")

    def test_worker_backend_when_remote_workers_on(self) -> None:
        with patch.dict(
            "os.environ",
            {"APP_REMOTE_WORKERS": "1", "TRIPO_DEV_MOCK": "0", "TRIPO_API_KEY": ""},
            clear=False,
        ):
            import app.tripo_core as tripo_core

            self.assertEqual(tripo_core.resolve_tripo_backend(), "worker_triposr")


if __name__ == "__main__":
    unittest.main()
