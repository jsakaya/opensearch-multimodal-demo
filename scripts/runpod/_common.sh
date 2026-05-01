#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATE_DIR="$REPO_ROOT/.runpod"
API="https://rest.runpod.io/v1"
mkdir -p "$STATE_DIR"

require_env() {
  local var="$1"
  if [[ -z "${!var:-}" ]]; then
    echo "missing required env: $var" >&2
    return 1
  fi
}

api() {
  local method="$1"; shift
  local path="$1"; shift
  local body="${1:-}"
  if [[ -n "$body" ]]; then
    curl -fsS -X "$method" "$API$path" \
      -H "Authorization: Bearer $RUNPOD_API_KEY" \
      -H "Content-Type: application/json" \
      -d "$body"
  else
    curl -fsS -X "$method" "$API$path" \
      -H "Authorization: Bearer $RUNPOD_API_KEY"
  fi
}
