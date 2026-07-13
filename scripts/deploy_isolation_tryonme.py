"""Deploy isolation (PicPolish) APIs + deps to tryonme.net."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1]
HOST = "49.12.204.14"
REMOTE = "/opt/polygraphics-backend"

UPLOAD_PATHS = [
    # Isolation
    "app/isolation_api.py",
    "app/isolation_db.py",
    "app/isolation_inference.py",
    "app/isolation_mask.py",
    "app/isolation_models.py",
    "app/isolation_quota.py",
    "app/isolation_service.py",
    "app/isolation_train.py",
    "scripts/isolation_train_worker.py",
    "tests/test_isolation_api.py",
    # Wiring + shared deps used by isolation
    "app/main.py",
    "app/jobs_db.py",
    "app/quota_db.py",
    "app/user_quota.py",
    "app/job_access.py",
    "worker/remote_worker.py",
    "requirements.txt",
    ".env.example",
    "README.md",
    "openapi.json",
]


def _ensure_remote_dir(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    parts = remote_dir.strip("/").split("/")
    cur = ""
    for part in parts:
        cur = f"{cur}/{part}"
        try:
            sftp.stat(cur)
        except OSError:
            sftp.mkdir(cur)


def main() -> int:
    password = os.environ.get("SSH_PASS")
    if not password:
        print("Set SSH_PASS", file=sys.stderr)
        return 1

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username="root", password=password, timeout=25, allow_agent=False, look_for_keys=False)
    sftp = client.open_sftp()

    for rel in UPLOAD_PATHS:
        local = ROOT / rel
        if not local.is_file():
            print("SKIP missing", rel)
            continue
        remote = f"{REMOTE}/{rel.replace(chr(92), '/')}"
        _ensure_remote_dir(sftp, str(Path(remote).parent).replace("\\", "/"))
        sftp.put(str(local), remote)
        print("uploaded", rel)
    sftp.close()

    cmds = [
        f"mkdir -p /var/lib/polygraphics/datasets/isolation /var/lib/polygraphics/models/isolation /var/lib/polygraphics/uploads/isolation",
        f"cd {REMOTE} && .venv/bin/pip install -q rembg onnxruntime opencv-python-headless",
        f"grep -q '^APP_FREE_ISOLATION_PER_MONTH=' {REMOTE}/.env || echo 'APP_FREE_ISOLATION_PER_MONTH=50' >> {REMOTE}/.env",
        f"grep -q '^ISOLATION_TRAIN_ON_API=' {REMOTE}/.env || echo 'ISOLATION_TRAIN_ON_API=0' >> {REMOTE}/.env",
        f"grep -q '^ISOLATION_TRAIN_DEV_MOCK=' {REMOTE}/.env || echo 'ISOLATION_TRAIN_DEV_MOCK=0' >> {REMOTE}/.env",
        f"cd {REMOTE} && .venv/bin/python -c \"from app.isolation_api import router; print('isolation import OK', len(router.routes))\"",
        f"cd {REMOTE} && .venv/bin/python scripts/export_openapi.py",
        "systemctl restart polygraphics-api",
        "sleep 3 && curl -sf https://tryonme.net/polygraph/health",
        "curl -sf https://tryonme.net/polygraph/isolation/health",
        "curl -s -o /dev/null -w '%{http_code}' https://tryonme.net/polygraph/isolation/predict",
        "curl -s -o /dev/null -w '%{http_code}' https://tryonme.net/polygraph/openapi.json",
        "systemctl is-active polygraphics-api",
    ]
    for cmd in cmds:
        _, stdout, stderr = client.exec_command(cmd, get_pty=True, timeout=300)
        out = (stdout.read() + stderr.read()).decode("utf-8", errors="replace").strip()
        print("---", cmd[:90])
        print(out[-1500:] if out else "(no output)")

    client.close()
    print("DEPLOY DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
