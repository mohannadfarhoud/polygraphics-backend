#!/usr/bin/env bash
# API-only install on Ubuntu (split deploy: remote GPU workers run reconstruction).
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/mohannadfarhoud/polygraphics-backend.git}"
BRANCH="${BRANCH:-feature/split-deploy-installers}"
INSTALL_DIR="${INSTALL_DIR:-/opt/polygraphics-backend}"
DATA_DIR="${DATA_DIR:-/var/lib/polygraphics}"
PUBLIC_BASE="${PUBLIC_BASE:-http://127.0.0.1/polygraph}"
WORKER_TOKEN="${WORKER_TOKEN:-}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run as root (or with sudo)." >&2
  exit 1
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
  python3 python3-venv python3-dev python3-pip git nginx curl \
  build-essential libgl1 libglib2.0-0 libgomp1

mkdir -p "$DATA_DIR"/uploads "$DATA_DIR"/output "$DATA_DIR"/config "$DATA_DIR"/data

if [[ ! -d "$INSTALL_DIR/.git" ]]; then
  git clone --branch "$BRANCH" --depth 1 "$REPO_URL" "$INSTALL_DIR"
else
  git -C "$INSTALL_DIR" fetch origin "$BRANCH"
  git -C "$INSTALL_DIR" checkout "$BRANCH"
  git -C "$INSTALL_DIR" pull --ff-only origin "$BRANCH" || true
fi

cd "$INSTALL_DIR"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
python -m pip install -r requirements.txt

if [[ -z "$WORKER_TOKEN" ]]; then
  WORKER_TOKEN="$(openssl rand -base64 32 | tr -d '/+=' | head -c 43)"
fi

PUBLIC_BASE="${PUBLIC_BASE%/}"

cat > "$INSTALL_DIR/.env" <<EOF
APP_HOST=0.0.0.0
APP_PORT=8000
APP_RELOAD=false
APP_WORKERS=1
APP_ROOT_DIR=${DATA_DIR}
APP_ROOT_PATH=/polygraph
APP_REMOTE_WORKERS=true
APP_WORKER_TOKEN=${WORKER_TOKEN}
APP_PUBLIC_BASE_URL=${PUBLIC_BASE}
APP_MODEL_BASE_URL=${PUBLIC_BASE}/output
APP_UPLOADS_BASE_URL=${PUBLIC_BASE}/uploads
APP_CORS_ORIGINS=*
EOF

cat > /etc/systemd/system/polygraphics-api.service <<EOF
[Unit]
Description=polyGraphics API (remote workers)
After=network.target

[Service]
Type=simple
WorkingDirectory=${INSTALL_DIR}
EnvironmentFile=${INSTALL_DIR}/.env
ExecStart=${INSTALL_DIR}/.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=5
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/nginx/sites-available/polygraphics <<'NGINX'
server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;
    client_max_body_size 250M;

    location ^~ /polygraph/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }

    location ^~ /output/ {
        proxy_pass http://127.0.0.1:8000/output/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        client_max_body_size 250M;
    }

    location ^~ /uploads/ {
        proxy_pass http://127.0.0.1:8000/uploads/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        client_max_body_size 250M;
    }
}
NGINX

rm -f /etc/nginx/sites-enabled/default
ln -sf /etc/nginx/sites-available/polygraphics /etc/nginx/sites-enabled/polygraphics
nginx -t

systemctl daemon-reload
systemctl enable polygraphics-api
systemctl restart polygraphics-api
systemctl enable nginx
systemctl restart nginx

echo ""
echo "=== polyGraphics API installed ==="
echo "Install dir:  ${INSTALL_DIR}"
echo "Data dir:     ${DATA_DIR}"
echo "Public base:  ${PUBLIC_BASE}"
echo "Health:       ${PUBLIC_BASE%/polygraph}/polygraph/health  (or curl http://127.0.0.1:8000/health)"
echo ""
echo "Copy to GPU worker .env.worker:"
echo "POLYGRAPH_API_BASE=${PUBLIC_BASE}"
echo "POLYGRAPH_WORKER_TOKEN=${WORKER_TOKEN}"
echo ""

curl -sf "http://127.0.0.1:8000/health" && echo " (uvicorn health OK)" || echo "WARN: health check failed — see journalctl -u polygraphics-api"
