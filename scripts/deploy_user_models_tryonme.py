"""Deploy per-user models + quota changes to tryonme.net API server."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1]
HOST = "49.12.204.14"
REMOTE = "/opt/polygraphics-backend"

UPLOAD_PATHS = [
    "app/quota_db.py",
    "app/user_quota.py",
    "app/job_access.py",
    "app/jobs_db.py",
    "app/job_models.py",
    "app/job_manager.py",
    "app/main.py",
    "requirements.txt",
    ".env.example",
    "tests/test_user_models.py",
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
    for rel in UPLOAD_PATHS:
        local = ROOT / rel
        remote = f"{REMOTE}/{rel.replace(chr(92), '/')}"
        sftp.put(str(local), remote)
        print("uploaded", rel)
    sftp.close()

    cmds = [
        f"grep -q '^APP_FREE_MODELS_PER_MONTH=' {REMOTE}/.env || echo 'APP_FREE_MODELS_PER_MONTH=10' >> {REMOTE}/.env",
        f"grep -q '^APP_REQUIRE_AUTH_FOR_JOBS=' {REMOTE}/.env || echo 'APP_REQUIRE_AUTH_FOR_JOBS=1' >> {REMOTE}/.env",
        f"cd {REMOTE} && .venv/bin/python scripts/export_openapi.py",
        f"cd {REMOTE} && APP_ROOT_DIR=/tmp/pg-user-models APP_DATABASE_PATH=/tmp/pg-user-models/db.sqlite APP_JWT_SECRET=test-jwt-secret-long APP_GOOGLE_CLIENT_ID=test APP_FREE_MODELS_PER_MONTH=10 .venv/bin/python -m unittest tests.test_user_models -v 2>&1 | tail -20",
        "systemctl restart polygraphics-api",
        "sleep 2 && curl -sf https://tryonme.net/polygraph/health",
        "curl -s -o /dev/null -w '%{http_code}' https://tryonme.net/polygraph/jobs",
    ]
    for cmd in cmds:
        _, stdout, stderr = client.exec_command(cmd, get_pty=True, timeout=180)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        print("---", cmd[:70], "---")
        print((out + err).strip()[-1500:])

    client.close()
    print("DEPLOY DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
