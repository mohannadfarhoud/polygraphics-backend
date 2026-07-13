"""Deploy incremental isolation training to tryonme.net."""
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
    "app/isolation_inference.py",
    "app/isolation_service.py",
    "app/isolation_api.py",
    "app/isolation_db.py",
    "app/isolation_models.py",
    "app/main.py",
    "worker/remote_worker.py",
    "README.md",
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
    cmds = [
        f"cd {REMOTE} && .venv/bin/python -c \"from app.isolation_finetune import torch_available; print('torch_api', torch_available())\"",
        f"cd {REMOTE} && .venv/bin/python scripts/export_openapi.py",
        "systemctl restart polygraphics-api",
        "sleep 3 && curl -sf https://tryonme.net/polygraph/isolation/health",
        "systemctl is-active polygraphics-api",
    ]
    for cmd in cmds:
        _, stdout, stderr = client.exec_command(cmd, get_pty=True, timeout=180)
        out = (stdout.read() + stderr.read()).decode("utf-8", errors="replace").strip()
        print("---", cmd[:90])
        print(out[-1000:] if out else "(no output)")
    client.close()
    print("DEPLOY DONE — pull on GPU worker and restart")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
