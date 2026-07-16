"""Clear all isolation models/datasets/train jobs on tryonme (fresh start)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import paramiko

HOST = "49.12.204.14"
DATA = "/var/lib/polygraphics"

REMOTE_PY = r'''
import sqlite3
import shutil
from pathlib import Path

data = Path("/var/lib/polygraphics")
db = data / "data" / "jobs.sqlite"
conn = sqlite3.connect(str(db))
for table in ("isolation_models", "isolation_train_jobs", "isolation_upload_events", "isolation_datasets"):
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    if not row:
        print(f"skip missing {table}")
        continue
    n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    conn.execute(f"DELETE FROM {table}")
    print(f"cleared {table}: {n}")
conn.commit()
conn.close()

for rel in ("models/isolation", "datasets/isolation", "uploads/isolation"):
    p = data / rel
    if p.exists():
        shutil.rmtree(p)
        print(f"removed {p}")
    p.mkdir(parents=True, exist_ok=True)
    print(f"recreated {p}")
print("OK")
'''


def main() -> int:
    password = os.environ.get("SSH_PASS")
    if not password:
        print("Set SSH_PASS", file=sys.stderr)
        return 1

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username="root", password=password, timeout=25, allow_agent=False, look_for_keys=False)

    sftp = client.open_sftp()
    with sftp.file("/tmp/clear_isolation.py", "w") as rf:
        rf.write(REMOTE_PY)
    sftp.close()

    cmds = [
        "python3 /tmp/clear_isolation.py",
        "systemctl restart polygraphics-api",
        "sleep 3 && curl -sf https://tryonme.net/polygraph/isolation/health",
        "curl -sf https://tryonme.net/polygraph/isolation/models; echo",
        "curl -s -o /dev/null -w 'active_http=%{http_code}\\n' https://tryonme.net/polygraph/isolation/models/active",
    ]
    for cmd in cmds:
        _, stdout, stderr = client.exec_command(cmd, get_pty=True, timeout=120)
        out = (stdout.read() + stderr.read()).decode("utf-8", errors="replace").strip()
        print("---", cmd)
        print(out[-1200:] if out else "(no output)")

    client.close()
    print("FRESH START DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
