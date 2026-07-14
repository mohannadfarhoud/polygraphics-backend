#!/bin/bash
set -e
cd /opt/polygraphics-backend
grep -q '^ISOLATION_TRAINER_PASSWORD=' .env || echo 'ISOLATION_TRAINER_PASSWORD=devtek2026' >> .env
set -a
# shellcheck disable=SC1091
source ./.env
set +a
.venv/bin/python -c "from pathlib import Path; import os; from app.isolation_auth import ensure_trainer_user; print('trainer=', ensure_trainer_user(Path(os.environ.get('APP_DATABASE_PATH','/var/lib/polygraphics/data/jobs.sqlite')))['email'])"
systemctl restart polygraphics-api
sleep 4
echo -n "noauth_ds="
curl -s -o /dev/null -w "%{http_code}" -X POST https://tryonme.net/polygraph/isolation/datasets \
  -H "Content-Type: application/json" -d '{"name":"x"}'
echo
TOK=$(curl -sf -X POST https://tryonme.net/polygraph/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"trainer","password":"devtek2026"}')
TOKEN=$(printf '%s' "$TOK" | .venv/bin/python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
echo "got_token=${#TOKEN}"
echo -n "trainer_ds="
curl -s -o /dev/null -w "%{http_code}" -X POST https://tryonme.net/polygraph/isolation/datasets \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" -d '{"name":"trainer-check"}'
echo
systemctl is-active polygraphics-api
