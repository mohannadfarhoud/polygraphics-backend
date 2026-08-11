"""Deploy isolation training open to any authenticated user."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib import request

import paramiko

ROOT = Path(__file__).resolve().parents[1]
HOST = "49.12.204.14"
REMOTE = "/opt/polygraphics-backend"
PUBLIC_BASE = "https://tryonme.net/polygraph"


def _json_request(method: str, path: str, *, body: dict | None = None, token: str | None = None) -> tuple[int, dict]:
    headers: dict[str, str] = {}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body).encode()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = request.Request(PUBLIC_BASE + path, data=data, headers=headers, method=method)
    with request.urlopen(req, timeout=30) as response:
        return response.status, json.loads(response.read().decode())


def main() -> int:
    password = os.environ.get("SSH_PASS")
    if not password:
        print("Set SSH_PASS", file=sys.stderr)
        return 1

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username="root", password=password, timeout=25, allow_agent=False, look_for_keys=False)
    sftp = client.open_sftp()
    sftp.put(str(ROOT / "app/isolation_api.py"), f"{REMOTE}/app/isolation_api.py")
    sftp.close()
    print("uploaded app/isolation_api.py")

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

    # Register a throwaway user and verify dataset create works (not 403).
    email = f"traincheck-{int(__import__('time').time())}@test.local"
    status, reg = _json_request(
        "POST",
        "/auth/register",
        body={"email": email, "password": "secret12345", "display_name": "TrainCheck"},
    )
    if status != 200:
        raise RuntimeError(f"register failed: {status}")
    status, ds = _json_request(
        "POST",
        "/isolation/datasets",
        body={"name": "any-user-check"},
        token=reg["access_token"],
    )
    if status != 201:
        raise RuntimeError(f"dataset create failed: {status}")
    print("any_user_train_ok", ds.get("dataset_id"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
