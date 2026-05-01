#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [[ ! -f "$STATE_DIR/ssh" ]]; then
  echo "no SSH info; run scripts/runpod/up.sh first" >&2
  exit 1
fi
read -r IP PORT < "$STATE_DIR/ssh"

ssh -o StrictHostKeyChecking=accept-new -p "$PORT" root@"$IP" "$@"
