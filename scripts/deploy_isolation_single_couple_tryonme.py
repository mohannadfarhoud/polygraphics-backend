"""Deploy single-couple pair upload support to tryonme."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1]
HOST = "49.12.204.14"
REMOTE = "/opt/polygraphics-backend"

FILES = [
    "app/isolation_api.py",
    "app/isolation_service.py",
    "app/isolation_models.py",
    "scripts/verify_isolation_single_couple.py",
]


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
        remote = f"{REMOTE}/{rel}"
        # ensure scripts dir exists
        if "/" in rel:
            try:
                sftp.stat(f"{REMOTE}/{rel.rsplit('/', 1)[0]}")
            except OSError:
                sftp.mkdir(f"{REMOTE}/{rel.rsplit('/', 1)[0]}")
        sftp.put(str(ROOT / rel), remote)
        print("uploaded", rel)
    sftp.close()

    remote_check = (
        "set -e; "
        "systemctl restart polygraphics-api; "
        "sleep 5; "
        "cd /opt/polygraphics-backend; "
        ".venv/bin/python scripts/verify_isolation_single_couple.py; "
        "systemctl is-active polygraphics-api"
    )
    _, stdout, stderr = client.exec_command(remote_check, get_pty=True, timeout=180)
    out = (stdout.read() + stderr.read()).decode("utf-8", errors="replace")
    print(out)
    client.close()
    if "single_couple_ok" not in out:
        print("VERIFY FAILED", file=sys.stderr)
        return 1
    print("DEPLOY DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
