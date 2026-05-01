#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

require_env RUNPOD_API_KEY
if [[ ! -f "$STATE_DIR/pod-id" ]]; then
  echo "no pod registered" >&2
  exit 0
fi

POD_ID="$(cat "$STATE_DIR/pod-id")"
api DELETE "/pods/$POD_ID" >/dev/null
rm -f "$STATE_DIR/pod-id" "$STATE_DIR/ssh"
echo "terminated pod $POD_ID" >&2
