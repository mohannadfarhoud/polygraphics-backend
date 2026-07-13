import gc
import importlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient


def _png_bytes(bgr: np.ndarray, alpha: np.ndarray | None = None) -> bytes:
    if alpha is not None:
        bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = alpha
        ok, enc = cv2.imencode(".png", bgra)
    else:
        ok, enc = cv2.imencode(".png", bgr)
    assert ok
    return enc.tobytes()


class IsolationApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self._root, ignore_errors=True))
        for sub in ("data", "output", "uploads", "config", "datasets", "models"):
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
                "APP_JWT_SECRET": "test-jwt-secret",
                "APP_REMOTE_WORKERS": "0",
                "ISOLATION_TRAIN_ON_API": "1",
                "ISOLATION_TRAIN_DEV_MOCK": "1",
                "ISOLATION_BASE_ONNX_PATH": "",
            },
            clear=False,
        )
        self._env.start()
        import app.main as main_mod

        importlib.reload(main_mod)
        self.client = TestClient(main_mod.app)

        reg = self.client.post(
            "/auth/register",
            json={"email": "iso@test.local", "password": "secret123", "display_name": "Iso"},
        )
        self.assertEqual(reg.status_code, 200, reg.text)
        self.token = reg.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def tearDown(self) -> None:
        self._env.stop()
        gc.collect()

    def _before_after_pair(self) -> tuple[bytes, bytes]:
        before = np.full((64, 64, 3), (40, 80, 120), dtype=np.uint8)
        after = np.full((64, 64, 3), 255, dtype=np.uint8)
        after[16:48, 16:48] = (40, 80, 120)
        return _png_bytes(before), _png_bytes(after)

    def test_dataset_pairs_and_mask_generation(self) -> None:
        ds = self.client.post(
            "/isolation/datasets",
            json={"name": "picpolish-v1"},
            headers=self.headers,
        )
        self.assertEqual(ds.status_code, 201, ds.text)
        dataset_id = ds.json()["dataset_id"]
        b, a = self._before_after_pair()
        files = [
            ("before", ("b0.png", b, "image/png")),
            ("after", ("a0.png", a, "image/png")),
        ]
        data = {"indices": "0"}
        up = self.client.post(
            f"/isolation/datasets/{dataset_id}/pairs",
            headers=self.headers,
            data=data,
            files=files,
        )
        self.assertEqual(up.status_code, 200, up.text)
        detail = self.client.get(f"/isolation/datasets/{dataset_id}", headers=self.headers)
        self.assertEqual(detail.status_code, 200)
        self.assertEqual(detail.json()["pair_count"], 1)
        mask_path = self._root / "datasets" / "isolation" / dataset_id / "pairs" / "0" / "mask.png"
        self.assertTrue(mask_path.is_file())

    @patch("app.isolation_train._export_rembg_onnx")
    def test_train_activate_predict(self, mock_export) -> None:
        def _fake_export(_base: str, dest: Path) -> None:
            # minimal onnx-like blob for import path; predict is mocked below
            dest.write_bytes(b"\x08\x03" + b"\x00" * 1024)

        mock_export.side_effect = _fake_export

        ds = self.client.post("/isolation/datasets", json={"name": "train-set"}, headers=self.headers)
        dataset_id = ds.json()["dataset_id"]
        b, a = self._before_after_pair()
        self.client.post(
            f"/isolation/datasets/{dataset_id}/pairs",
            headers=self.headers,
            data={"indices": "0,1"},
            files=[
                ("before", ("b0.png", b, "image/png")),
                ("before", ("b1.png", b, "image/png")),
                ("after", ("a0.png", a, "image/png")),
                ("after", ("a1.png", a, "image/png")),
            ],
        )
        train = self.client.post(
            "/isolation/train",
            headers=self.headers,
            json={
                "dataset_id": dataset_id,
                "epochs": 2,
                "val_split": 0.5,
                "force_min_pairs": True,
            },
        )
        self.assertEqual(train.status_code, 202, train.text)
        job_id = train.json()["job_id"]

        final = None
        for _ in range(80):
            got = self.client.get(f"/isolation/train/{job_id}", headers=self.headers)
            final = got.json()
            if final["status"] in ("completed", "failed"):
                break
        assert final is not None
        self.assertEqual(final["status"], "completed", final)
        model_id = final["model_id"]
        self.assertTrue(model_id)

        act = self.client.post(f"/isolation/models/{model_id}/activate", headers=self.headers)
        self.assertEqual(act.status_code, 200)

        fake_rgba = _png_bytes(
            np.full((32, 32, 3), 100, dtype=np.uint8),
            alpha=np.full((32, 32), 200, dtype=np.uint8),
        )
        with patch("app.isolation_service.predict_isolated_png", return_value=(fake_rgba, fake_rgba, 12)):
            pred = self.client.post(
                "/isolation/predict",
                headers=self.headers,
                files={"file": ("test.png", b, "image/png")},
            )
        self.assertEqual(pred.status_code, 200, pred.text)
        self.assertEqual(pred.headers.get("content-type"), "image/png")
        self.assertTrue(pred.content.startswith(b"\x89PNG"))

    def test_health_and_auth(self) -> None:
        h = self.client.get("/isolation/health")
        self.assertEqual(h.status_code, 200)
        r = self.client.post("/isolation/datasets", json={"name": "x"})
        self.assertEqual(r.status_code, 401)


if __name__ == "__main__":
    unittest.main()
