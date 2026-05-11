#!/usr/bin/env bash
# Start the polygraphics-backend on Colab/Linux and expose it via a free
# tunnel. Tries Cloudflare Quick Tunnel first, then falls back to
# localhost.run (SSH) if Cloudflare's try.cloudflare.com API is broken
# (it sometimes returns HTML 500 → "error code: 1101 invalid character 'e'").
#
# Usage:
#   bash scripts/colab_serve.sh                # tries cloudflared, then lhr
#   TUNNEL=cloudflared  bash scripts/colab_serve.sh
#   TUNNEL=localhost.run bash scripts/colab_serve.sh
#   TUNNEL=serveo       bash scripts/colab_serve.sh
set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
PORT="${PORT:-8000}"
LOG="${LOG:-/tmp/polygraph.log}"
PIDFILE="${PIDFILE:-/tmp/polygraph.pid}"
TUNNEL_LOG="${TUNNEL_LOG:-/tmp/tunnel.log}"
TUNNEL_PIDFILE="${TUNNEL_PIDFILE:-/tmp/tunnel.pid}"
TUNNEL="${TUNNEL:-auto}"        # auto | cloudflared | localhost.run | serveo

cd "$REPO_DIR"

# ---------- 1. uvicorn ----------
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

for _ in $(seq 1 60); do
  sleep 1
  if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "    API is up (http://127.0.0.1:$PORT)."
    break
  fi
done

if ! curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
  echo "!! API failed to start. Last 80 log lines:" >&2
  tail -n 80 "$LOG" >&2 || true
  exit 1
fi

# ---------- 2. tunnel helpers ----------
extract_url() {
  # Args: <log-file>. Echoes first https://...trycloudflare/lhr.life/serveo URL.
  grep -oE 'https://[a-z0-9-]+\.(trycloudflare\.com|lhr\.life|serveo\.net)' "$1" 2>/dev/null \
    | head -n1
}

stop_old_tunnel() {
  if [ -f "$TUNNEL_PIDFILE" ] && kill -0 "$(cat "$TUNNEL_PIDFILE")" 2>/dev/null; then
    kill "$(cat "$TUNNEL_PIDFILE")" 2>/dev/null || true
    sleep 1
  fi
  rm -f "$TUNNEL_PIDFILE" "$TUNNEL_LOG"
  pkill -f cloudflared 2>/dev/null || true
}

try_cloudflared() {
  if ! command -v cloudflared >/dev/null 2>&1; then
    echo "==> Installing cloudflared..."
    if command -v sudo >/dev/null 2>&1; then SUDO=sudo; else SUDO=; fi
    $SUDO wget --no-verbose -O /usr/local/bin/cloudflared \
      https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    $SUDO chmod +x /usr/local/bin/cloudflared
  fi
  echo "==> Starting Cloudflare Quick Tunnel..."
  nohup cloudflared tunnel --no-autoupdate --url "http://localhost:$PORT" \
    > "$TUNNEL_LOG" 2>&1 &
  echo $! > "$TUNNEL_PIDFILE"
  for _ in $(seq 1 30); do
    sleep 1
    URL="$(extract_url "$TUNNEL_LOG")"
    [ -n "${URL:-}" ] && { echo "$URL"; return 0; }
    if grep -qE '(error code: 1101|Internal Server Error|invalid character)' "$TUNNEL_LOG" 2>/dev/null; then
      echo "    cloudflared: try.cloudflare.com is misbehaving (1101)." >&2
      return 1
    fi
  done
  echo "    cloudflared: no URL after 30s." >&2
  return 1
}

try_lhr() {
  echo "==> Starting localhost.run tunnel (no signup, SSH-based)..."
  nohup ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=60 -o ExitOnForwardFailure=yes \
    -R "80:localhost:$PORT" nokey@localhost.run \
    > "$TUNNEL_LOG" 2>&1 &
  echo $! > "$TUNNEL_PIDFILE"
  for _ in $(seq 1 30); do
    sleep 1
    URL="$(extract_url "$TUNNEL_LOG")"
    [ -n "${URL:-}" ] && { echo "$URL"; return 0; }
  done
  echo "    localhost.run: no URL after 30s." >&2
  return 1
}

try_serveo() {
  echo "==> Starting serveo tunnel..."
  nohup ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=60 -o ExitOnForwardFailure=yes \
    -R "80:localhost:$PORT" serveo.net \
    > "$TUNNEL_LOG" 2>&1 &
  echo $! > "$TUNNEL_PIDFILE"
  for _ in $(seq 1 30); do
    sleep 1
    URL="$(extract_url "$TUNNEL_LOG")"
    [ -n "${URL:-}" ] && { echo "$URL"; return 0; }
  done
  echo "    serveo: no URL after 30s." >&2
  return 1
}

# ---------- 3. pick a tunnel ----------
URL=""
case "$TUNNEL" in
  cloudflared)   stop_old_tunnel; URL="$(try_cloudflared || true)" ;;
  localhost.run) stop_old_tunnel; URL="$(try_lhr          || true)" ;;
  serveo)        stop_old_tunnel; URL="$(try_serveo       || true)" ;;
  auto|*)
    stop_old_tunnel
    URL="$(try_cloudflared || true)"
    [ -z "$URL" ] && { stop_old_tunnel; URL="$(try_lhr    || true)"; }
    [ -z "$URL" ] && { stop_old_tunnel; URL="$(try_serveo || true)"; }
    ;;
esac

if [ -z "$URL" ]; then
  echo
  echo "!! No tunnel could be established. Tunnel log tail:" >&2
  tail -n 60 "$TUNNEL_LOG" >&2 || true
  exit 1
fi

echo
echo "================================================================"
echo " API URL: $URL"
echo " Try:    $URL/health"
echo " Try:    $URL/swagger"
echo " Try:    $URL/server/status"
echo "================================================================"
echo
echo "Tunnel pid : $(cat "$TUNNEL_PIDFILE" 2>/dev/null) (log: $TUNNEL_LOG)"
echo "uvicorn pid: $(cat "$PIDFILE" 2>/dev/null) (log: $LOG)"
echo
echo "Both run in the background. Stop with:"
echo "  kill \$(cat $TUNNEL_PIDFILE) \$(cat $PIDFILE)"
echo
echo "If the URL stops working, re-run this script — it will rotate."
