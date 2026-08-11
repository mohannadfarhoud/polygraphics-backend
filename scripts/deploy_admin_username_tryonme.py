"""Deploy admin username rename (trainer -> admin) to tryonme."""
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
    "app/isolation_auth.py",
    "app/auth_db.py",
]


def _json_request(method: str, path: str, *, body: dict | None = None) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    data = json.dumps(body).encode() if body is not None else None
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

    remote = r"""
set -e
cd /opt/polygraphics-backend
.venv/bin/python -c "from pathlib import Path; from app.isolation_auth import ensure_trainer_user; user=ensure_trainer_user(Path('/var/lib/polygraphics/data/jobs.sqlite')); print('admin_user=', user['email'], user['name'])"
systemctl restart polygraphics-api
sleep 4
systemctl is-active polygraphics-api
"""
    _, stdout, stderr = client.exec_command(remote, get_pty=True, timeout=120)
    out = (stdout.read() + stderr.read()).decode("utf-8", errors="replace")
    print(out)
    client.close()
    if "active" not in out:
        return 1

    status, login = _json_request("POST", "/auth/login", body={"email": "admin", "password": "devtek2026"})
    if status != 200:
        raise RuntimeError(f"admin login failed: {status}")
    print("admin_login_ok", login["user"]["email"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
