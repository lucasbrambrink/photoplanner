#!/usr/bin/env bash
# PhotoPlanner / LightFinder local auto-sync + preview server daemon.
#
# Watches `origin/main` on GitHub every POLL_SECONDS (default 10s):
#   - Fast-forwards clean `main` automatically when new commits arrive
#   - Only rebuilds the Rust `photoplanner` binary if `raycaster/` changed
#   - Only restarts `uvicorn` if Python or Rust files changed (`web/` serves live!)
#   - Never overwrites dirty local edits (logs + optional macOS notification instead)
#
# Usage:
#   ./scripts/dev-sync.sh          # Start uvicorn + auto-sync loop in foreground
#   ./scripts/dev-sync.sh --once   # Single pull + conditional Rust rebuild + restart check

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PATH="$HOME/.cargo/bin:$PATH"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8420}"
BRANCH="${BRANCH:-main}"
POLL_SECONDS="${POLL_SECONDS:-10}"
ONCE_MODE=0
if [[ "${1:-}" == "--once" ]]; then
  ONCE_MODE=1
fi

UVICORN_BIN="$ROOT_DIR/.venv/bin/uvicorn"
if [[ ! -x "$UVICORN_BIN" ]]; then
  echo "[dev-sync] ERROR: $UVICORN_BIN not found." >&2
  exit 1
fi

SERVER_PID=""
LAST_DIRTY_WARN_SHA=""

log() {
  printf '[dev-sync %s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

notify_mac() {
  local title="$1"
  local msg="$2"
  if command -v osascript >/dev/null 2>&1; then
    osascript -e "display notification \"${msg//\"/\\\"}\" with title \"${title//\"/\\\"}\"" >/dev/null 2>&1 || true
  fi
}

stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    log "Stopping uvicorn (pid $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 0.25
    done
    kill -9 "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}

start_server() {
  stop_server
  # Clear any stray uvicorn bound to this port before starting
  local existing_pids
  existing_pids="$(pgrep -f "uvicorn api_server:app --host ${HOST} --port ${PORT}" || true)"
  if [[ -n "$existing_pids" ]]; then
    log "Stopping pre-existing uvicorn on ${HOST}:${PORT} ($existing_pids)..."
    kill $existing_pids 2>/dev/null || true
    sleep 0.8
  fi

  log "Starting uvicorn on http://${HOST}:${PORT} ..."
  "$UVICORN_BIN" api_server:app --host "$HOST" --port "$PORT" &
  SERVER_PID=$!

  for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      log "ERROR: uvicorn exited prematurely."
      return 1
    fi
    if curl -fsS "http://${HOST}:${PORT}/api/health" >/dev/null 2>&1; then
      log "Server healthy at http://${HOST}:${PORT} (pid $SERVER_PID)"
      return 0
    fi
    sleep 0.5
  done
  log "WARNING: server started (pid $SERVER_PID) but /api/health did not respond yet."
}

rebuild_rust() {
  log "Rust sources changed — rebuilding raycaster release binary..."
  if cargo build --release --manifest-path "$ROOT_DIR/raycaster/Cargo.toml"; then
    log "Rust release binary rebuilt successfully."
    return 0
  else
    log "ERROR: Rust build failed — keeping existing binary."
    notify_mac "PhotoPlanner Sync" "Rust release build failed; check dev-sync logs."
    return 1
  fi
}

sync_once() {
  local current_branch
  current_branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '')"
  if [[ "$current_branch" != "$BRANCH" ]]; then
    return 0
  fi

  if ! git fetch --quiet origin "$BRANCH" 2>/dev/null; then
    return 0
  fi

  local local_sha remote_sha
  local_sha="$(git rev-parse HEAD)"
  remote_sha="$(git rev-parse "origin/$BRANCH")"

  if [[ "$local_sha" == "$remote_sha" ]]; then
    return 0
  fi

  # Skip if local working tree has uncommitted tracked edits
  if ! git diff --quiet || ! git diff --cached --quiet; then
    if [[ "$LAST_DIRTY_WARN_SHA" != "$remote_sha" ]]; then
      LAST_DIRTY_WARN_SHA="$remote_sha"
      log "New origin/${BRANCH} commit (${remote_sha:0:7}), but local tracked edits exist — skipping auto-pull."
      notify_mac "PhotoPlanner Sync" "New commit ${remote_sha:0:7} on GitHub, but local changes block auto-pull."
    fi
    return 0
  fi

  log "Syncing ${local_sha:0:7} -> ${remote_sha:0:7} ..."
  if ! git pull --ff-only --quiet origin "$BRANCH" 2>/dev/null; then
    if ! git pull --rebase --quiet origin "$BRANCH" 2>/dev/null; then
      git rebase --abort >/dev/null 2>&1 || true
      if [[ "$LAST_DIRTY_WARN_SHA" != "$remote_sha" ]]; then
        LAST_DIRTY_WARN_SHA="$remote_sha"
        log "WARNING: rebase onto origin/${BRANCH} (${remote_sha:0:7}) conflicted and was aborted."
        notify_mac "PhotoPlanner Sync" "Auto-pull conflict on ${remote_sha:0:7} — please resolve manually."
      fi
      return 1
    fi
  fi

  local new_sha changed needs_rust=0 needs_restart=0
  new_sha="$(git rev-parse HEAD)"
  changed="$(git diff --name-only "${local_sha}..${new_sha}")"
  log "Updated to $(git log -1 --oneline)"

  if grep -q '^raycaster/' <<< "$changed"; then
    needs_rust=1
    needs_restart=1
  fi
  if grep -qE '(^api_server\.py$|^server\.py$|^planner/)' <<< "$changed"; then
    needs_restart=1
  fi
  if grep -q '^requirements\.txt$' <<< "$changed"; then
    log "requirements.txt changed — installing updated Python dependencies..."
    "$ROOT_DIR/.venv/bin/pip" install -q -r "$ROOT_DIR/requirements.txt" || true
    needs_restart=1
  fi

  if [[ "$needs_rust" -eq 1 ]]; then
    rebuild_rust || true
  fi

  if [[ "$ONCE_MODE" -eq 1 ]]; then
    if [[ "$needs_restart" -eq 1 ]]; then
      log "Backend changed — restarting server..."
      pkill -f "uvicorn api_server:app --host ${HOST} --port ${PORT}" 2>/dev/null || true
      sleep 0.8
      nohup "$UVICORN_BIN" api_server:app --host "$HOST" --port "$PORT" >/tmp/photoplanner-uvicorn.log 2>&1 &
      log "Restarted uvicorn in background (logs: /tmp/photoplanner-uvicorn.log)"
    else
      log "Only frontend/docs changed — no server restart needed (refresh browser)."
    fi
  else
    if [[ "$needs_restart" -eq 1 ]]; then
      start_server
    else
      log "Only frontend/docs changed — live files updated (refresh browser)."
    fi
  fi
}

if [[ "$ONCE_MODE" -eq 1 ]]; then
  sync_once
  exit 0
fi

trap 'stop_server; exit 0' INT TERM EXIT

# Pull any pending updates first, then launch server
sync_once || true
start_server

log "Watching origin/${BRANCH} every ${POLL_SECONDS}s (Ctrl-C to stop)..."
while true; do
  sleep "$POLL_SECONDS"
  # Auto-recover server if it unexpectedly died
  if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
    log "uvicorn exited unexpectedly — restarting..."
    start_server || true
  fi
  sync_once || true
done
