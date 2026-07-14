"""Deploy trainer-auth for isolation training uploads to tryonme."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1]
HOST = "49.12.204.14"
REMOTE = "/opt/polygraphics-backend"

FILES = [
    "app/isolation_auth.py",
    "app/isolation_api.py",
    "app/auth_service.py",
    "app/auth_db.py",
    "tests/test_isolation_api.py",
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

    remote_check = r'''
set -e
grep -q '^ISOLATION_TRAINER_PASSWORD=' /opt/polygraphics-backend/.env \
  || echo 'ISOLATION_TRAINER_PASSWORD=devtek2026' >> /opt/polygraphics-backend/.env
cd /opt/polygraphics-backend
.venv/bin/python - <<'PY'
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()
from app.isolation_auth import ensure_trainer_user
db = Path(os.getenv("APP_DATABASE_PATH", "/var/lib/polygraphics/data/jobs.sqlite"))
print("trainer=", ensure_trainer_user(db)["email"])
PY
systemctl restart polygraphics-api
sleep 3
curl -s -o /dev/null -w "noauth_ds=%{http_code}\n" -X POST https://tryonme.net/polygraph/isolation/datasets \
  -H "Content-Type: application/json" -d '{"name":"x"}'
TOKEN=$(curl -sf -X POST https://tryonme.net/polygraph/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"trainer","password":"devtek2026"}' | .venv/bin/python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
curl -s -o /dev/null -w "trainer_ds=%{http_code}\n" -X POST https://tryonme.net/polygraph/isolation/datasets \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{"name":"trainer-check"}'
systemctl is-active polygraphics-api
'''
    _, stdout, stderr = client.exec_command(remote_check, get_pty=True, timeout=180)
    print((stdout.read() + stderr.read()).decode("utf-8", errors="replace"))
    client.close()
    print("DEPLOY DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
