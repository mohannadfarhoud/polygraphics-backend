"""Deploy isolation ONNX export fix to tryonme.net."""
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
    for rel in (
        "app/isolation_train.py",
        "app/isolation_inference.py",
        "requirements.txt",
    ):
        sftp.put(str(ROOT / rel), f"{REMOTE}/{rel}")
        print("uploaded", rel)
    sftp.close()

    cmds = [
        f'cd {REMOTE} && .venv/bin/pip install -q "rembg[cpu]" onnxruntime',
        (
            f"cd {REMOTE} && .venv/bin/python -c \""
            "from pathlib import Path; import tempfile; "
            "from app.isolation_train import _export_rembg_onnx; "
            "p=Path(tempfile.mkdtemp())/'model.onnx'; "
            "_export_rembg_onnx('isnet-general-use', p); "
            "print('export_ok', p.stat().st_size)\""
        ),
        "systemctl restart polygraphics-api",
        "sleep 2 && curl -sf https://tryonme.net/polygraph/isolation/health",
        "systemctl is-active polygraphics-api",
    ]
    for cmd in cmds:
        _, stdout, stderr = client.exec_command(cmd, get_pty=True, timeout=600)
        out = (stdout.read() + stderr.read()).decode("utf-8", errors="replace").strip()
        print("---", cmd[:100])
        print(out[-1500:] if out else "(no output)")

    client.close()
    print("DEPLOY DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
