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
                "APP_JWT_SECRET": "test-jwt-secret-at-least-32-bytes-long!!",
                "APP_REMOTE_WORKERS": "0",
                "ISOLATION_TRAIN_ON_API": "1",
                "ISOLATION_TRAIN_DEV_MOCK": "1",
                "ISOLATION_BASE_ONNX_PATH": "",
                "ISOLATION_TRAINER_PASSWORD": "devtek2026",
            },
            clear=False,
        )
        self._env.start()
        import app.main as main_mod

        importlib.reload(main_mod)
        self.client = TestClient(main_mod.app)

        login = self.client.post(
            "/auth/login",
            json={"email": "admin", "password": "devtek2026"},
        )
        self.assertEqual(login.status_code, 200, login.text)
        self.token = login.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def tearDown(self) -> None:
        self._env.stop()
        gc.collect()

    def _before_after_pair(self, seed: int = 0) -> tuple[bytes, bytes]:
        before = np.full((64, 64, 3), (40 + seed, 80, 120), dtype=np.uint8)
        after = np.full((64, 64, 3), 255, dtype=np.uint8)
        after[16:48, 16:48] = (40 + seed, 80, 120)
        return _png_bytes(before), _png_bytes(after)

    def test_single_couple_upload_without_indices(self) -> None:
        ds = self.client.post(
            "/isolation/datasets",
            json={"name": "single-couple"},
            headers=self.headers,
        )
        self.assertEqual(ds.status_code, 201, ds.text)
        dataset_id = ds.json()["dataset_id"]
        b, a = self._before_after_pair()
        up = self.client.post(
            f"/isolation/datasets/{dataset_id}/pairs",
            headers=self.headers,
            files=[
                ("before", ("b0.png", b, "image/png")),
                ("after", ("a0.png", a, "image/png")),
            ],
        )
        self.assertEqual(up.status_code, 200, up.text)
        body = up.json()
        self.assertEqual(body["uploaded"], 1)
        self.assertEqual(body["pair_indices"], [0])
        detail = self.client.get(f"/isolation/datasets/{dataset_id}", headers=self.headers)
        self.assertEqual(detail.json()["pair_count"], 1)

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

    def test_statistics_track_contributor_success_and_failure(self) -> None:
        ds = self.client.post(
            "/isolation/datasets",
            json={"name": "statistics"},
            headers=self.headers,
        )
        dataset_id = ds.json()["dataset_id"]
        b, a = self._before_after_pair()

        success = self.client.post(
            f"/isolation/datasets/{dataset_id}/pairs",
            headers=self.headers,
            files=[
                ("before", ("b0.png", b, "image/png")),
                ("after", ("a0.png", a, "image/png")),
            ],
        )
        self.assertEqual(success.status_code, 200, success.text)

        failure = self.client.post(
            f"/isolation/datasets/{dataset_id}/pairs",
            headers=self.headers,
            files=[
                ("before", ("b1.png", b, "image/png")),
                ("after", ("a1.png", a, "image/png")),
                ("after", ("a2.png", a, "image/png")),
            ],
        )
        self.assertEqual(failure.status_code, 400, failure.text)

        stats = self.client.get(
            "/isolation/statistics",
            params={"dataset_id": dataset_id},
            headers=self.headers,
        )
        self.assertEqual(stats.status_code, 200, stats.text)
        body = stats.json()
        self.assertEqual(body["datasets"]["current_pairs"], 1)
        self.assertEqual(body["uploads"]["upload_attempts"], 2)
        self.assertEqual(body["uploads"]["submitted_photos"], 5)
        self.assertEqual(body["uploads"]["successful_photos"], 2)
        self.assertEqual(body["uploads"]["unsuccessful_photos"], 3)
        self.assertEqual(body["uploads"]["successful_pairs"], 1)
        self.assertEqual(body["uploads"]["unsuccessful_pairs"], 2)
        self.assertEqual(len(body["contributors"]), 1)
        self.assertEqual(body["contributors"][0]["email"], "admin@polygraph.local")

    def test_reject_duplicate_pairs_and_admin_pair_list(self) -> None:
        ds = self.client.post(
            "/isolation/datasets",
            json={"name": "dup-check"},
            headers=self.headers,
        )
        dataset_id = ds.json()["dataset_id"]
        b, a = self._before_after_pair()
        first = self.client.post(
            f"/isolation/datasets/{dataset_id}/pairs",
            headers=self.headers,
            files=[
                ("before", ("b0.png", b, "image/png")),
                ("after", ("a0.png", a, "image/png")),
            ],
        )
        self.assertEqual(first.status_code, 200, first.text)
        self.assertEqual(first.json()["uploaded"], 1)

        dup = self.client.post(
            f"/isolation/datasets/{dataset_id}/pairs",
            headers=self.headers,
            files=[
                ("before", ("b0-again.png", b, "image/png")),
                ("after", ("a0-again.png", a, "image/png")),
            ],
        )
        self.assertEqual(dup.status_code, 400, dup.text)
        self.assertIn("duplicate", dup.json()["detail"].lower())

        pairs = self.client.get("/isolation/pairs", headers=self.headers)
        self.assertEqual(pairs.status_code, 200, pairs.text)
        body = pairs.json()
        self.assertEqual(body["total"], 1)
        self.assertEqual(body["counts"]["total_pairs"], 1)
        self.assertEqual(body["counts"]["admin_pairs"], 1)
        self.assertEqual(len(body["pairs"]), 1)
        self.assertEqual(body["pairs"][0]["uploaded_by_email"], "admin@polygraph.local")
        self.assertTrue(body["pairs"][0]["is_admin_uploader"])
        self.assertTrue(body["pairs"][0]["uploaded_at"])
        self.assertTrue(body["pairs"][0]["before_url"])
        self.assertTrue(body["pairs"][0]["after_url"])
        before = self.client.get(body["pairs"][0]["before_url"], headers=self.headers)
        self.assertEqual(before.status_code, 200, before.text)
        self.assertTrue(before.content.startswith(b"\x89PNG"))
        after = self.client.get(
            body["pairs"][0]["after_url"],
            params={"access_token": self.token},
        )
        self.assertEqual(after.status_code, 200, after.text)

        stats = self.client.get("/isolation/statistics", headers=self.headers)
        self.assertEqual(stats.status_code, 200, stats.text)
        self.assertEqual(stats.json()["pairs"]["admin_pairs"], 1)

    @patch("app.isolation_train._export_rembg_onnx")
    def test_train_activate_predict(self, mock_export) -> None:
        def _fake_export(_base: str, dest: Path) -> None:
            dest.write_bytes(b"\x08\x03" + b"\x00" * 1024)

        mock_export.side_effect = _fake_export

        ds = self.client.post(
            "/isolation/datasets",
            json={"name": "train-set"},
            headers=self.headers,
        )
        dataset_id = ds.json()["dataset_id"]
        b0, a0 = self._before_after_pair(0)
        b1, a1 = self._before_after_pair(1)
        self.client.post(
            f"/isolation/datasets/{dataset_id}/pairs",
            headers=self.headers,
            data={"indices": "0,1"},
            files=[
                ("before", ("b0.png", b0, "image/png")),
                ("before", ("b1.png", b1, "image/png")),
                ("after", ("a0.png", a0, "image/png")),
                ("after", ("a1.png", a1, "image/png")),
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
                files={"file": ("test.png", b0, "image/png")},
            )
        self.assertEqual(pred.status_code, 200, pred.text)
        self.assertEqual(pred.headers.get("content-type"), "image/png")
        self.assertTrue(pred.content.startswith(b"\x89PNG"))

    def test_logged_in_user_can_train(self) -> None:
        h = self.client.get("/isolation/health")
        self.assertEqual(h.status_code, 200)
        # Upload/train requires authentication.
        r = self.client.post("/isolation/datasets", json={"name": "x"})
        self.assertEqual(r.status_code, 401, r.text)
        # Predict stays public.
        pred = self.client.post(
            "/isolation/predict",
            files={"file": ("t.png", b"\x89PNG\r\n\x1a\n", "image/png")},
        )
        self.assertIn(pred.status_code, (503, 400))
        self.assertNotEqual(pred.status_code, 401)
        # Admin login exposes is_admin for the UI admin panel.
        me = self.client.get("/auth/me", headers=self.headers)
        self.assertEqual(me.status_code, 200, me.text)
        self.assertTrue(me.json().get("is_admin"))
        # Any registered user may create datasets and train.
        reg = self.client.post(
            "/auth/register",
            json={"email": "other@test.local", "password": "secret123", "display_name": "Other"},
        )
        self.assertEqual(reg.status_code, 200, reg.text)
        self.assertFalse(reg.json()["user"].get("is_admin"))
        other_headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}
        allowed = self.client.post(
            "/isolation/datasets",
            json={"name": "user-dataset"},
            headers=other_headers,
        )
        self.assertEqual(allowed.status_code, 201, allowed.text)
        # Admin panel (statistics) is admin-only.
        denied = self.client.get("/isolation/statistics", headers=other_headers)
        self.assertEqual(denied.status_code, 403, denied.text)
        ok = self.client.get("/isolation/statistics", headers=self.headers)
        self.assertEqual(ok.status_code, 200, ok.text)

    def test_dedupe_and_retrain_all(self) -> None:
        ds = self.client.post(
            "/isolation/datasets",
            json={"name": "dedupe-set"},
            headers=self.headers,
        )
        dataset_id = ds.json()["dataset_id"]
        b, a = self._before_after_pair(3)

        # Bypass upload reject to simulate legacy duplicate couples already on disk/meta.
        from app.isolation_api import get_isolation_service

        svc = get_isolation_service()
        meta = svc._load_meta(dataset_id)
        for idx in (0, 1):
            pair_dir = svc._dataset_dir(dataset_id) / "pairs" / str(idx)
            pair_dir.mkdir(parents=True, exist_ok=True)
            (pair_dir / "before.png").write_bytes(b)
            (pair_dir / "after.png").write_bytes(a)
            from app.isolation_mask import generate_and_save_mask

            generate_and_save_mask(
                before_path=str(pair_dir / "before.png"),
                after_path=str(pair_dir / "after.png"),
                mask_path=str(pair_dir / "mask.png"),
            )
            content_hash = svc._pair_content_hash(b, a)
            meta.setdefault("pairs", []).append(
                {
                    "index": idx,
                    "before": f"pairs/{idx}/before.png",
                    "after": f"pairs/{idx}/after.png",
                    "mask": f"pairs/{idx}/mask.png",
                    "content_hash": content_hash if idx == 0 else content_hash,
                    "uploaded_by_user_id": None,
                }
            )
            if idx == 0:
                from app import isolation_db

                isolation_db.insert_pair(
                    svc.db_path,
                    dataset_id=dataset_id,
                    pair_index=0,
                    content_hash=content_hash,
                    uploaded_by_user_id=None,
                    before_rel=f"pairs/0/before.png",
                    after_rel=f"pairs/0/after.png",
                    mask_rel=f"pairs/0/mask.png",
                )
            else:
                # Second row: force into meta/disk only (same hash can't insert twice).
                pass
        svc._save_meta(dataset_id, meta)
        from app import isolation_db

        isolation_db.update_dataset(svc.db_path, dataset_id, pair_count=2)

        from app.isolation_train import _list_pairs

        listed = _list_pairs(svc._dataset_dir(dataset_id))
        self.assertEqual(len(listed), 1)

        reg = self.client.post(
            "/auth/register",
            json={"email": "u2@test.local", "password": "secret123", "display_name": "U2"},
        )
        self.assertEqual(reg.status_code, 200, reg.text)
        other_headers = {"Authorization": f"Bearer {reg.json()['access_token']}"}
        denied = self.client.post(
            "/isolation/retrain-all",
            headers=other_headers,
            json={"dataset_id": dataset_id, "epochs": 1, "force_min_pairs": True},
        )
        self.assertEqual(denied.status_code, 403, denied.text)

        retrain = self.client.post(
            "/isolation/retrain-all",
            headers=self.headers,
            json={
                "dataset_id": dataset_id,
                "epochs": 1,
                "force_min_pairs": True,
                "purge_duplicates": True,
                "grow_active": False,
            },
        )
        self.assertEqual(retrain.status_code, 202, retrain.text)
        body = retrain.json()
        self.assertEqual(body["dataset_id"], dataset_id)
        self.assertEqual(body["dedupe"]["kept"], 1)
        self.assertEqual(body["dedupe"]["removed"], 1)
        self.assertIn(body["train"]["status"], ("queued", "running", "completed"))

        detail = self.client.get(f"/isolation/datasets/{dataset_id}", headers=self.headers)
        self.assertEqual(detail.json()["pair_count"], 1)


if __name__ == "__main__":
    unittest.main()
