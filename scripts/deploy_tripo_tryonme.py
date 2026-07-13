"""Deploy Tripo worker-backend fix to tryonme.net."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1]
HOST = "49.12.204.14"
REMOTE = "/opt/polygraphics-backend"


def main() -> int:
    password = os.environ.get("SSH_PASS")
    if not password:
        print("Set SSH_PASS", file=sys.stderr)
        return 1

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username="root", password=password, timeout=25, allow_agent=False, look_for_keys=False)
    sftp = client.open_sftp()

    upload_paths = [
        "app/tripo_api.py",
        "app/tripo_core.py",
        "app/tripo_db.py",
        "app/tripo_service.py",
        "app/tripo_triposr.py",
        "app/main.py",
        "worker/remote_worker.py",
        ".env.example",
    ]
    for rel in upload_paths:
        local = ROOT / rel
        remote = f"{REMOTE}/{rel.replace(chr(92), '/')}"
        sftp.put(str(local), remote)
        print("uploaded", rel)
    sftp.close()

    cmds = [
        f"grep -q '^APP_PUBLIC_BASE_URL=' {REMOTE}/.env || echo 'APP_PUBLIC_BASE_URL=https://tryonme.net/polygraph' >> {REMOTE}/.env",
        f"cd {REMOTE} && .venv/bin/python scripts/export_openapi.py",
        "systemctl restart polygraphics-api",
        "sleep 2 && curl -sf https://tryonme.net/polygraph/health",
    ]
    for cmd in cmds:
        _, stdout, stderr = client.exec_command(cmd, get_pty=True, timeout=120)
        print("---", cmd[:80])
        print((stdout.read() + stderr.read()).decode("utf-8", errors="replace").strip()[-1200:])

    client.close()
    print("DEPLOY DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
