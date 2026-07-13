import gc
import importlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


class UserModelsApiTests(unittest.TestCase):
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
                "APP_JWT_SECRET": "test-jwt-secret-long-enough-for-hmac",
                "APP_GOOGLE_CLIENT_ID": "test-google-client-id",
                "APP_FREE_MODELS_PER_MONTH": "10",
            },
            clear=False,
        )
        self._env.start()
        import app.main as main_mod

        importlib.reload(main_mod)
        self.client = TestClient(main_mod.app)
        self.main = main_mod

    def tearDown(self) -> None:
        self._env.stop()
        gc.collect()

    def _login(self, *, sub: str, email: str, token: str) -> tuple[str, str]:
        with patch("app.auth_service.verify_google_id_token") as mock_verify:
            mock_verify.return_value = {
                "sub": sub,
                "email": email,
                "email_verified": True,
            }
            r = self.client.post("/auth/google", json={"id_token": token})
        self.assertEqual(r.status_code, 200, r.text)
        access_token = r.json()["access_token"]
        me = self.client.get("/auth/me", headers={"Authorization": f"Bearer {access_token}"})
        self.assertEqual(me.status_code, 200, me.text)
        return access_token, me.json()["user_id"]

    def test_jobs_require_auth(self) -> None:
        r = self.client.get("/jobs")
        self.assertEqual(r.status_code, 401)

    def test_user_sees_only_own_models(self) -> None:
        token_a, user_a = self._login(sub="google-user-1", email="user@example.com", token="token-a")
        headers_a = {"Authorization": f"Bearer {token_a}"}

        job_a = "job-a"
        self.main.job_manager.create_job_pending(job_a, 2, owner_user_id=user_a)
        (self._root / "output" / f"{job_a}.glb").write_bytes(b"glb")

        _token_b, user_b = self._login(sub="google-user-2", email="other@example.com", token="token-b")
        job_b = "job-b"
        self.main.job_manager.create_job_pending(job_b, 2, owner_user_id=user_b)
        (self._root / "output" / f"{job_b}.glb").write_bytes(b"glb")

        models_a = self.client.get("/users/me/models", headers=headers_a).json()
        self.assertEqual(len(models_a), 1)
        self.assertEqual(models_a[0]["job_id"], job_a)
        self.assertIn("/jobs/job-a/download", models_a[0]["download_url"])

    def test_monthly_quota_blocks_eleventh_start(self) -> None:
        token, user_id = self._login(sub="google-user-quota", email="quota@example.com", token="token-q")
        headers = {"Authorization": f"Bearer {token}"}

        for i in range(10):
            jid = f"quota-job-{i}"
            job_dir = self._root / "uploads" / jid
            job_dir.mkdir(parents=True, exist_ok=True)
            (job_dir / "input_000.png").write_bytes(b"x")
            (job_dir / "input_001.png").write_bytes(b"y")
            self.main.job_manager.create_job_pending(jid, 2, owner_user_id=user_id)
            started = self.main.job_manager.start_job(jid, owner_user_id=user_id)
            self.assertEqual(started.status.value, "queued")

        quota = self.client.get("/users/me/quota", headers=headers).json()
        self.assertEqual(quota["used"], 10)
        self.assertEqual(quota["remaining"], 0)

        extra = "quota-job-11"
        self.main.job_manager.create_job_pending(extra, 1, owner_user_id=user_id)
        r = self.client.post(f"/jobs/{extra}/start", headers=headers)
        self.assertEqual(r.status_code, 429, r.text)


if __name__ == "__main__":
    unittest.main()
