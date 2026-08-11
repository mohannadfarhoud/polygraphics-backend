"""Deploy isolation training statistics API to tryonme."""
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

FILES = [
    "app/isolation_api.py",
    "app/isolation_db.py",
    "app/isolation_models.py",
    "app/isolation_service.py",
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
    for rel in FILES:
        sftp.put(str(ROOT / rel), f"{REMOTE}/{rel}")
        print("uploaded", rel)
    sftp.close()

    _, stdout, stderr = client.exec_command(
        "systemctl restart polygraphics-api; sleep 4; systemctl is-active polygraphics-api",
        get_pty=True,
        timeout=120,
    )
    output = (stdout.read() + stderr.read()).decode("utf-8", errors="replace")
    print(output)
    client.close()
    if "active" not in output:
        return 1

    status, login = _json_request(
        "POST",
        "/auth/login",
        body={"email": "admin", "password": "devtek2026"},
    )
    if status != 200:
        raise RuntimeError(f"trainer login failed: {status}")
    status, statistics = _json_request("GET", "/isolation/statistics", token=login["access_token"])
    if status != 200 or "uploads" not in statistics or "contributors" not in statistics:
        raise RuntimeError(f"statistics verification failed: {status} {statistics}")
    print("statistics_ok", json.dumps(statistics, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
