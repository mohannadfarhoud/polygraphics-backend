import gc
import importlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


class AuthApiTests(unittest.TestCase):
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
                "APP_JWT_SECRET": "test-jwt-secret",
                "APP_GOOGLE_CLIENT_ID": "test-google-client-id",
                "APP_GOOGLE_CLIENT_SECRET": "test-google-client-secret",
                "APP_PUBLIC_BASE_URL": "http://testserver/polygraph",
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

    def test_auth_me_requires_token(self) -> None:
        r = self.client.get("/auth/me")
        self.assertEqual(r.status_code, 401)

    @patch("app.auth_service.verify_google_id_token")
    def test_google_sign_in_creates_user_and_returns_jwt(self, mock_verify) -> None:
        mock_verify.return_value = {
            "sub": "google-sub-1",
            "email": "alice@example.com",
            "email_verified": True,
            "name": "Alice",
            "picture": "https://example.com/a.png",
        }
        r = self.client.post("/auth/google", json={"credential": "fake-google-token"})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertIn("access_token", body)
        self.assertEqual(body["user"]["email"], "alice@example.com")

        me = self.client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {body['access_token']}"},
        )
        self.assertEqual(me.status_code, 200, me.text)
        self.assertEqual(me.json()["email"], "alice@example.com")

    @patch("app.auth_service.verify_google_id_token")
    def test_google_sign_in_is_idempotent_login(self, mock_verify) -> None:
        mock_verify.return_value = {
            "sub": "google-sub-2",
            "email": "bob@example.com",
            "email_verified": True,
        }
        first = self.client.post("/auth/google", json={"id_token": "token-1"}).json()
        second = self.client.post("/auth/google", json={"id_token": "token-2"}).json()
        self.assertEqual(first["user"]["user_id"], second["user"]["user_id"])

    @patch("app.auth_service.verify_google_id_token")
    def test_user_calibration_accepts_bearer_token(self, mock_verify) -> None:
        mock_verify.return_value = {
            "sub": "google-sub-3",
            "email": "carol@example.com",
            "email_verified": True,
        }
        auth = self.client.post("/auth/google", json={"id_token": "token-3"}).json()
        token = auth["access_token"]
        put = self.client.put(
            "/users/me/try-on-calibration",
            json={"left_offset": {"vertical": 0.02, "depth": 0, "lateral": 0}},
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(put.status_code, 200, put.text)
        got = self.client.get(
            "/users/me/try-on-calibration",
            headers={"Authorization": f"Bearer {token}"},
        )
        self.assertEqual(got.status_code, 200, got.text)
        self.assertAlmostEqual(got.json()["left_offset"]["vertical"], 0.02, places=4)


    def test_email_register_and_login(self) -> None:
        reg = self.client.post(
            "/auth/register",
            json={"email": "local@example.com", "password": "secret123", "name": "Local User"},
        )
        self.assertEqual(reg.status_code, 200, reg.text)
        body = reg.json()
        self.assertEqual(body["user"]["email"], "local@example.com")
        self.assertIn("access_token", body)

        bad = self.client.post(
            "/auth/login",
            json={"email": "local@example.com", "password": "wrong"},
        )
        self.assertEqual(bad.status_code, 401)

        ok = self.client.post(
            "/auth/login",
            json={"email": "local@example.com", "password": "secret123"},
        )
        self.assertEqual(ok.status_code, 200, ok.text)
        self.assertEqual(ok.json()["user"]["user_id"], body["user"]["user_id"])

        dup = self.client.post(
            "/auth/register",
            json={"email": "local@example.com", "password": "secret123"},
        )
        self.assertEqual(dup.status_code, 409)


if __name__ == "__main__":
    unittest.main()
