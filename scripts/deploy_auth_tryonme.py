"""Deploy Google auth files to tryonme.net API server."""
from __future__ import annotations

import os
import secrets
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
        "app/auth_db.py",
        "app/auth_models.py",
        "app/auth_service.py",
        "app/auth_deps.py",
        "app/auth_api.py",
        "app/main.py",
        "app/jobs_db.py",
        "app/try_on_api.py",
        "requirements.txt",
        ".env.example",
        "tests/test_auth.py",
        "tests/test_try_on_config.py",
    ]
    for rel in upload_paths:
        local = ROOT / rel
        remote = f"{REMOTE}/{rel.replace(chr(92), '/')}"
        remote_dir = str(Path(remote).parent).replace("\\", "/")
        try:
            sftp.stat(remote_dir)
        except OSError:
            pass
        sftp.put(str(local), remote)
        print("uploaded", rel)

    sftp.close()

    jwt_secret = secrets.token_urlsafe(48)
    cmds = [
        f"cd {REMOTE} && .venv/bin/pip install -q PyJWT google-auth",
        f"grep -q '^APP_JWT_SECRET=' {REMOTE}/.env || echo 'APP_JWT_SECRET={jwt_secret}' >> {REMOTE}/.env",
        f"grep -q '^APP_AUTH_FRONTEND_URL=' {REMOTE}/.env || echo 'APP_AUTH_FRONTEND_URL=https://tryonme.net' >> {REMOTE}/.env",
        f"grep -q '^APP_AUTH_FRONTEND_CALLBACK=' {REMOTE}/.env || echo 'APP_AUTH_FRONTEND_CALLBACK=https://tryonme.net/auth/callback' >> {REMOTE}/.env",
        f"cd {REMOTE} && .venv/bin/python scripts/export_openapi.py",
        "systemctl restart polygraphics-api",
        "sleep 2 && curl -sf https://tryonme.net/polygraph/health",
        "curl -sfI https://tryonme.net/polygraph/auth/me | head -3",
        f"grep -E 'APP_JWT_SECRET|APP_GOOGLE|APP_AUTH' {REMOTE}/.env",
    ]
    for cmd in cmds:
        _, stdout, stderr = client.exec_command(cmd, get_pty=True, timeout=120)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        print("---", cmd[:70], "---")
        print((out + err).strip()[-1200:])

    client.close()
    print("DEPLOY DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
