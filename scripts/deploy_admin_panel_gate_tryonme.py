"""Deploy admin-panel gating (is_admin + require_admin) to tryonme."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from urllib import error, request

import paramiko

ROOT = Path(__file__).resolve().parents[1]
HOST = "49.12.204.14"
REMOTE = "/opt/polygraphics-backend"
PUBLIC_BASE = "https://tryonme.net/polygraph"

FILES = [
    "app/auth_models.py",
    "app/auth_service.py",
    "app/isolation_auth.py",
    "app/isolation_api.py",
    "app/main.py",
]


def _json_request(method: str, path: str, *, body: dict | None = None, token: str | None = None) -> tuple[int, dict]:
    headers: dict[str, str] = {}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = request.Request(PUBLIC_BASE + path, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=30) as response:
            raw = response.read().decode()
            return response.status, json.loads(raw) if raw else {}
    except error.HTTPError as exc:
        raw = exc.read().decode()
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {"detail": raw}
        return exc.code, payload


def main() -> int:
    password = os.environ.get("SSH_PASS")
    if not password:
        print("Set SSH_PASS", file=sys.stderr)
        return 1

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username="root", password=password, timeout=25, allow_agent=False, look_for_keys=False)
    sftp = client.open_sftp()
    for rel in FILES:
        sftp.put(str(ROOT / rel), f"{REMOTE}/{rel}")
        print("uploaded", rel)
    sftp.close()

    _, stdout, stderr = client.exec_command(
        "systemctl restart polygraphics-api; sleep 4; systemctl is-active polygraphics-api",
        get_pty=True,
        timeout=120,
    )
    out = (stdout.read() + stderr.read()).decode("utf-8", errors="replace")
    print(out)
    client.close()
    if "active" not in out:
        return 1

    status, admin_login = _json_request(
        "POST",
        "/auth/login",
        body={"email": "admin", "password": "devtek2026"},
    )
    if status != 200 or not admin_login.get("user", {}).get("is_admin"):
        raise RuntimeError(f"admin login/is_admin failed: {status} {admin_login}")

    email = f"user-{int(time.time())}@test.local"
    status, reg = _json_request(
        "POST",
        "/auth/register",
        body={"email": email, "password": "secret12345", "display_name": "User"},
    )
    if status != 200 or reg.get("user", {}).get("is_admin"):
        raise RuntimeError(f"register is_admin failed: {status} {reg}")

    status, _ = _json_request("GET", "/isolation/statistics", token=reg["access_token"])
    if status != 403:
        raise RuntimeError(f"expected 403 for non-admin statistics, got {status}")

    status, stats = _json_request("GET", "/isolation/statistics", token=admin_login["access_token"])
    if status != 200 or "uploads" not in stats:
        raise RuntimeError(f"admin statistics failed: {status} {stats}")

    print("admin_panel_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
