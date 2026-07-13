"""Deploy HTTPS + WebSocket nginx and certbot certs on tryonme.net (one-shot)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1]
HOST = "49.12.204.14"
USER = "root"


def read_local(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8").replace("\r\n", "\n")


def main() -> int:
    password = os.environ.get("SSH_PASS")
    if not password:
        print("Set SSH_PASS", file=sys.stderr)
        return 1

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=password, timeout=25, allow_agent=False, look_for_keys=False)

    def run(cmd: str, timeout: int = 120) -> tuple[int, str]:
        _, stdout, stderr = client.exec_command(cmd, get_pty=True, timeout=timeout)
        out = stdout.read().decode("utf-8", errors="replace")
        err = stderr.read().decode("utf-8", errors="replace")
        code = stdout.channel.recv_exit_status()
        return code, (out + err).strip()

    def write_remote(path: str, content: str) -> None:
        sftp = client.open_sftp()
        with sftp.file(path, "w") as f:
            f.write(content)
        sftp.close()

    # Phase 1: HTTP-only config for ACME (no PowerShell $ expansion issues)
    acme_http = """server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name tryonme.net www.tryonme.net api.tryonme.net;

    location ^~ /.well-known/acme-challenge/ {
        root /var/www/tryonme-ui;
        allow all;
    }

    location ^~ /polygraph/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    root /var/www/tryonme-ui;
    location / {
        try_files $uri $uri/ /index.html;
    }
}
"""
    write_remote("/etc/nginx/sites-available/polygraphics", acme_http)
    write_remote("/etc/nginx/conf.d/polygraph-ws-map.conf", read_local("deploy/nginx-ws-map.conf"))

    code, out = run("nginx -t 2>&1 && systemctl reload nginx 2>&1")
    print("nginx acme config:", code, out[-800:])
    if code != 0:
        client.close()
        return code

    cert_cmd = (
        "certbot certonly --webroot -w /var/www/tryonme-ui "
        "-d tryonme.net -d www.tryonme.net -d api.tryonme.net "
        "--non-interactive --agree-tos --register-unsafely-without-email 2>&1"
    )
    code, out = run(cert_cmd, timeout=180)
    print("certbot:", code)
    print(out[-2500:])
    if code != 0:
        client.close()
        return code

    run("test -f /etc/letsencrypt/ssl-dhparams.pem || openssl dhparam -out /etc/letsencrypt/ssl-dhparams.pem 2048")
    run(
        "test -f /etc/letsencrypt/options-ssl-nginx.conf || "
        "curl -sfL https://raw.githubusercontent.com/certbot/certbot/master/certbot-nginx/certbot_nginx/_internal/tls_configs/options-ssl-nginx.conf "
        "-o /etc/letsencrypt/options-ssl-nginx.conf"
    )

    https_conf = read_local("deploy/nginx-tryonme-https.conf")
    lines = https_conf.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("# --- HTTP"))
    write_remote("/etc/nginx/sites-available/polygraphics", "\n".join(lines[start:]) + "\n")

    code, out = run("nginx -t 2>&1 && systemctl reload nginx 2>&1")
    print("nginx https reload:", code, out[-800:])
    if code != 0:
        client.close()
        return code

    env_cmds = [
        "sed -i 's|^APP_PUBLIC_BASE_URL=.*|APP_PUBLIC_BASE_URL=https://tryonme.net/polygraph|' /opt/polygraphics-backend/.env",
        "sed -i 's|^APP_MODEL_BASE_URL=.*|APP_MODEL_BASE_URL=https://tryonme.net/polygraph/output|' /opt/polygraphics-backend/.env",
        "sed -i 's|^APP_UPLOADS_BASE_URL=.*|APP_UPLOADS_BASE_URL=https://tryonme.net/polygraph/uploads|' /opt/polygraphics-backend/.env",
        "sed -i 's|^APP_CORS_ORIGINS=.*|APP_CORS_ORIGINS=https://tryonme.net,https://www.tryonme.net,https://api.tryonme.net|' /opt/polygraphics-backend/.env",
        "systemctl restart polygraphics-api",
        "systemctl enable certbot.timer 2>/dev/null; systemctl start certbot.timer 2>/dev/null; true",
    ]
    for cmd in env_cmds:
        run(cmd)

    checks = [
        "curl -sfI https://tryonme.net/polygraph/health | head -5",
        "curl -sf https://tryonme.net/polygraph/health",
        "curl -sfI http://tryonme.net/ | head -3",
        "grep APP_PUBLIC /opt/polygraphics-backend/.env",
    ]
    for cmd in checks:
        _, out = run(cmd)
        print("---", cmd, "---")
        print(out)

    client.close()
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
