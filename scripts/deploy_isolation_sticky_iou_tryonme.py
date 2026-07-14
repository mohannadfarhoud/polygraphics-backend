"""Deploy sticky IoU val-split fix to tryonme API."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1]
HOST = "49.12.204.14"
REMOTE = "/opt/polygraphics-backend"

FILES = [
    "app/isolation_finetune.py",
    "app/isolation_train.py",
    "app/isolation_models.py",
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
        sftp.put(str(ROOT / rel), f"{REMOTE}/{rel}")
        print("uploaded", rel)
    sftp.close()
    _, stdout, stderr = client.exec_command(
        "systemctl restart polygraphics-api; sleep 3; systemctl is-active polygraphics-api; "
        "grep -n sticky_hash /opt/polygraphics-backend/app/isolation_finetune.py | head -3",
        get_pty=True,
        timeout=120,
    )
    print((stdout.read() + stderr.read()).decode("utf-8", errors="replace"))
    client.close()
    print("DEPLOY DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
