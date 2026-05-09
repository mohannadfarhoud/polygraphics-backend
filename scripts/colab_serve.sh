#!/usr/bin/env bash
# Start the polygraphics-backend on Colab/Linux and expose it via a
# free Cloudflare Quick Tunnel (no account, no token).
#
# Usage:
#   !bash scripts/colab_serve.sh        # foreground; tunnel URL streams to stdout
#
# Stop with Ctrl+C / Stop the cell. uvicorn is killed via the saved PID.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PORT="${PORT:-8000}"
LOG="${LOG:-/tmp/polygraph.log}"
PIDFILE="${PIDFILE:-/tmp/polygraph.pid}"

cd "$REPO_DIR"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "==> Installing cloudflared..."
  if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi
  $SUDO wget --no-verbose -O /usr/local/bin/cloudflared \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  $SUDO chmod +x /usr/local/bin/cloudflared
fi

# Stop any previously-started instance (rerunning the cell shouldn't stack ports).
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "==> Stopping previous uvicorn (pid $(cat "$PIDFILE"))..."
  kill "$(cat "$PIDFILE")" 2>/dev/null || true
  sleep 1
fi
rm -f "$PIDFILE"

echo "==> Starting uvicorn on 0.0.0.0:$PORT..."
nohup python -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT" \
  > "$LOG" 2>&1 &
echo $! > "$PIDFILE"

# Wait for /health to come up so the tunnel doesn't print before the API is ready.
for _ in $(seq 1 60); do
  sleep 1
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "    API is up (http://127.0.0.1:$PORT)."
    break
  fi
done

if ! curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "!! API failed to start. Last 50 log lines:" >&2
  tail -n 50 "$LOG" >&2 || true
  exit 1
fi

echo
echo "==> Opening a free Cloudflare Quick Tunnel."
echo "    Look for the line: 'https://<random>.trycloudflare.com'"
echo "    That's your public demo URL. Append /swagger to test."
echo

# `exec` lets cloudflared own the cell. Stopping the cell will tear it down;
# the next run of this script will restart uvicorn cleanly via $PIDFILE.
exec cloudflared tunnel --no-autoupdate --url "http://localhost:$PORT"
