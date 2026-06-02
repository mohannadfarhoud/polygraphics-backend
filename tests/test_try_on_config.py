import gc
import importlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.try_on_config import (
    ModelPoint3D,
    ModelTryOnConfigPut,
    TryOnConfigStore,
    merge_put,
    validate_profile_distance,
)


class TryOnValidationTests(unittest.TestCase):
    def test_profile_too_close_raises(self) -> None:
        hanger = ModelPoint3D(x=0, y=0.5, z=0)
        profile = ModelPoint3D(x=0.001, y=0.5, z=0)
        with self.assertRaises(ValueError):
            validate_profile_distance(hanger, profile)

    def test_merge_hanger_only_preserves_profile(self) -> None:
        existing = merge_put(
            "job-a",
            None,
            ModelTryOnConfigPut(
                hanger_point=ModelPoint3D(x=0, y=0.5, z=0),
                profile_point=ModelPoint3D(x=0.4, y=0.45, z=0),
            ),
            require_hanger=True,
        )
        updated = merge_put(
            "job-a",
            existing,
            ModelTryOnConfigPut(hanger_point=ModelPoint3D(x=0.02, y=0.48, z=-0.01)),
            require_hanger=False,
        )
        self.assertIsNotNone(updated.profile_point)
        self.assertAlmostEqual(updated.profile_point.x, 0.4, places=4)
        self.assertAlmostEqual(updated.hanger_point.x, 0.02, places=4)


class TryOnStoreTests(unittest.TestCase):
    def _store(self) -> tuple[TryOnConfigStore, Path]:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        db = tmp / "jobs.sqlite"
        return TryOnConfigStore(db), tmp

    def test_put_get_roundtrip(self) -> None:
        store, _tmp = self._store()
        saved = store.upsert(
            "job-1",
            ModelTryOnConfigPut(
                hanger_point=ModelPoint3D(x=0, y=0.5, z=0),
                profile_point=ModelPoint3D(x=0.4, y=0.45, z=0),
            ),
            require_hanger=True,
        )
        loaded = store.get("job-1")
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.hanger_point.x, saved.hanger_point.x)
        self.assertEqual(loaded.profile_point.x, 0.4)
        del store
        gc.collect()

    def test_hanger_only_does_not_clear_profile(self) -> None:
        store, _tmp = self._store()
        store.upsert(
            "job-2",
            ModelTryOnConfigPut(
                hanger_point=ModelPoint3D(x=0, y=0.5, z=0),
                profile_point=ModelPoint3D(x=0.35, y=0.42, z=0),
            ),
            require_hanger=True,
        )
        store.upsert_hanger_only("job-2", ModelPoint3D(x=0.1, y=0.48, z=0))
        loaded = store.get("job-2")
        assert loaded is not None
        self.assertAlmostEqual(loaded.hanger_point.x, 0.1, places=4)
        self.assertAlmostEqual(loaded.profile_point.x, 0.35, places=4)
        del store
        gc.collect()


class TryOnApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._root = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(self._root, ignore_errors=True))
        (self._root / "data").mkdir(parents=True, exist_ok=True)
        (self._root / "output").mkdir(parents=True, exist_ok=True)
        (self._root / "uploads").mkdir(parents=True, exist_ok=True)
        (self._root / "config").mkdir(parents=True, exist_ok=True)
        (self._root / "config" / "runtime_settings.json").write_text(
            json.dumps({"output_dir_name": "output"}),
            encoding="utf-8",
        )
        self._env = patch.dict(
            "os.environ",
            {
                "APP_ROOT_DIR": str(self._root),
                "APP_DATABASE_PATH": str(self._root / "data" / "test.sqlite"),
                "APP_TRY_ON_BIND_OWNER": "0",
            },
            clear=False,
        )
        self._env.start()
        import app.main as main_mod

        importlib.reload(main_mod)
        self.client = TestClient(main_mod.app)

    def _touch_model(self, job_id: str) -> None:
        (self._root / "output" / f"{job_id}.glb").write_bytes(b"glb")

    def tearDown(self) -> None:
        self._env.stop()
        gc.collect()

    def test_try_on_put_get_and_hanger_compat(self) -> None:
        self._touch_model("test-job")
        body = {
            "hanger_point": {"x": 0, "y": 0.5, "z": 0},
            "profile_point": {"x": 0.4, "y": 0.45, "z": 0},
            "jewelry_rotation": {"spin_deg": 0, "pitch_deg": 12, "roll_deg": -5},
            "api_vertical_flip": True,
            "jewelry_type": "drop",
        }
        put = self.client.put("/models/test-job/try-on", json=body)
        self.assertEqual(put.status_code, 200, put.text)
        get = self.client.get("/models/test-job/try-on")
        self.assertEqual(get.status_code, 200, get.text)
        self.assertEqual(get.json()["hanger_point"], body["hanger_point"])
        self.assertEqual(get.json()["api_vertical_flip"], True)

        hp_put = self.client.put(
            "/models/test-job/hanger-point",
            json={"x": 0.02, "y": 0.48, "z": -0.01},
        )
        self.assertEqual(hp_put.status_code, 200, hp_put.text)
        full = self.client.get("/models/test-job/try-on").json()
        self.assertAlmostEqual(full["hanger_point"]["x"], 0.02, places=4)
        self.assertAlmostEqual(full["profile_point"]["x"], 0.4, places=4)

        hp_get = self.client.get("/models/test-job/hanger-point")
        self.assertEqual(hp_get.status_code, 200)
        self.assertAlmostEqual(hp_get.json()["x"], 0.02, places=4)

    def test_profile_too_close_400(self) -> None:
        self._touch_model("test-job")
        r = self.client.put(
            "/models/test-job/try-on",
            json={
                "hanger_point": {"x": 0, "y": 0.5, "z": 0},
                "profile_point": {"x": 0, "y": 0.501, "z": 0},
            },
        )
        self.assertEqual(r.status_code, 400)
        self.assertIn("profile_point", r.json()["detail"].lower())

    def test_unknown_job_404(self) -> None:
        r = self.client.get("/models/no-such-job/try-on")
        self.assertEqual(r.status_code, 404)

    def test_user_calibration_requires_header(self) -> None:
        r = self.client.get("/users/me/try-on-calibration")
        self.assertEqual(r.status_code, 400)
        put = self.client.put(
            "/users/me/try-on-calibration",
            json={"left_offset": {"vertical": 0.01, "depth": 0, "lateral": 0}},
            headers={"X-User-Id": "user-1"},
        )
        self.assertEqual(put.status_code, 200, put.text)
        got = self.client.get(
            "/users/me/try-on-calibration",
            headers={"X-User-Id": "user-1"},
        )
        self.assertEqual(got.status_code, 200)
        self.assertAlmostEqual(got.json()["left_offset"]["vertical"], 0.01, places=4)

    def test_owner_mismatch_403(self) -> None:
        with patch.dict("os.environ", {"APP_TRY_ON_BIND_OWNER": "1"}, clear=False):
            import app.main as main_mod

            importlib.reload(main_mod)
            client = TestClient(main_mod.app)
            self._touch_model("owner-job")
            client.put(
                "/models/owner-job/try-on",
                json={"hanger_point": {"x": 0, "y": 0.5, "z": 0}},
                headers={"X-User-Id": "alice"},
            )
            r = client.get(
                "/models/owner-job/try-on",
                headers={"X-User-Id": "bob"},
            )
            self.assertEqual(r.status_code, 403)


if __name__ == "__main__":
    unittest.main()
