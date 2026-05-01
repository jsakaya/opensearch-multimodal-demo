#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [[ ! -f "$STATE_DIR/ssh" ]]; then
  echo "no SSH info; run scripts/runpod/up.sh first" >&2
  exit 1
fi
read -r IP PORT < "$STATE_DIR/ssh"

REMOTE_SCRIPT="$REPO_ROOT/scripts/runpod/full-power-demo-remote.sh"
LOCAL_PORT="${OPENLENS_LOCAL_PORT:-8787}"
CONTROL_PATH="$STATE_DIR/openlens-demo-tunnel.ctl"

ssh -o StrictHostKeyChecking=accept-new -p "$PORT" root@"$IP" \
  "OPENLENS_TARGET_DOCS=${OPENLENS_TARGET_DOCS:-10000} OPENLENS_COLPALI_BATCH_SIZE=${OPENLENS_COLPALI_BATCH_SIZE:-4} OPENLENS_COLPALI_MAX_BATCH=${OPENLENS_COLPALI_MAX_BATCH:-16} OPENLENS_AUTOTUNE_COLPALI=${OPENLENS_AUTOTUNE_COLPALI:-1} OPENLENS_QWEN_MAX_BATCH=${OPENLENS_QWEN_MAX_BATCH:-64} bash -s" \
  < "$REMOTE_SCRIPT"

if [[ "${OPENLENS_OPEN_TUNNEL:-1}" == "1" ]]; then
  if [[ -S "$CONTROL_PATH" ]] && ssh -S "$CONTROL_PATH" -O check root@"$IP" >/dev/null 2>&1; then
    :
  else
    rm -f "$CONTROL_PATH"
    ssh -fN -M -S "$CONTROL_PATH" \
      -o StrictHostKeyChecking=accept-new \
      -o ExitOnForwardFailure=yes \
      -p "$PORT" \
      -L "$LOCAL_PORT:127.0.0.1:8787" \
      root@"$IP"
  fi
  echo "browser URL: http://127.0.0.1:$LOCAL_PORT"
  echo "close tunnel: ssh -S $CONTROL_PATH -O exit root@$IP"
fi
