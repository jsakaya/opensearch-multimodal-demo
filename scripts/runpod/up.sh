#!/usr/bin/env bash
# Create a RunPod pod for OpenLens Qwen3-VL-Embedding encoding.
#
# Required:
#   RUNPOD_API_KEY, or a macOS Keychain item named runpod-api-key
#
# Optional:
#   RUNPOD_VOLUME_ID=t0ys2ffnll
#   RUNPOD_VOLUME_NAME=josephsakaya-unsloth-h100
#   RUNPOD_POD_NAME=openlens-qwen-h200
#   RUNPOD_GPU_ID="NVIDIA H200"
#   RUNPOD_DATA_CENTER_IDS=US-CA-2
#   RUNPOD_IMAGE=ghcr.io/jsakaya/openlens-qwen-encoder:latest
#   RUNPOD_PUBKEY=~/.ssh/id_ed25519.pub

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

load_runpod_key

NAME="${RUNPOD_POD_NAME:-openlens-qwen-h200}"
GPU="${RUNPOD_GPU_ID:-${RUNPOD_GPU:-NVIDIA H200}}"
DATA_CENTER_IDS="${RUNPOD_DATA_CENTER_IDS:-US-CA-2}"
IMAGE="${RUNPOD_IMAGE:-ghcr.io/jsakaya/openlens-qwen-encoder:latest}"
PUBKEY_FILE="${RUNPOD_PUBKEY:-$HOME/.ssh/id_ed25519.pub}"
[[ -f "$PUBKEY_FILE" ]] || { echo "no SSH public key at $PUBKEY_FILE" >&2; exit 1; }
PUBKEY="$(cat "$PUBKEY_FILE")"
VOLUME_ID="${RUNPOD_VOLUME_ID:-}"
if [[ -z "$VOLUME_ID" ]]; then
  VOLUME_NAME="${RUNPOD_VOLUME_NAME:-josephsakaya-unsloth-h100}"
  VOLUME_ID="$(runpodctl network-volume list -o json | jq -r --arg name "$VOLUME_NAME" '.[] | select(.name == $name) | .id' | head -1)"
fi
[[ -n "$VOLUME_ID" ]] || { echo "missing RUNPOD_VOLUME_ID and could not resolve RUNPOD_VOLUME_NAME" >&2; exit 1; }

if [[ -f "$STATE_DIR/pod-id" ]]; then
  echo "pod already registered: $(cat "$STATE_DIR/pod-id")" >&2
  exit 1
fi

ENV_JSON=$(PUBKEY="$PUBKEY" python3 <<'PY'
import json, os
print(json.dumps({
    "PUBLIC_KEY": os.environ["PUBKEY"],
    "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
    "OPENLENS_EMBEDDING_BACKEND": "qwen",
    "OPENLENS_QWEN_MODEL": "qwen8b",
    "OPENLENS_VECTOR_DIM": "4096",
    "OPENLENS_QWEN_BATCH_SIZE": os.environ.get("OPENLENS_QWEN_BATCH_SIZE", "16"),
    "OPENLENS_QWEN_MAX_FRAMES": os.environ.get("OPENLENS_QWEN_MAX_FRAMES", "64"),
    "OPENLENS_QWEN_FPS": os.environ.get("OPENLENS_QWEN_FPS", "1.0"),
    "OPENLENS_REQUIRE_OPENSEARCH": "1",
}))
PY
)

echo "creating $NAME ($GPU, data centers=$DATA_CENTER_IDS, image=$IMAGE, volume=$VOLUME_ID)..." >&2
RESPONSE="$(runpodctl pod create \
  --name "$NAME" \
  --image "$IMAGE" \
  --cloud-type SECURE \
  --gpu-id "$GPU" \
  --gpu-count 1 \
  --data-center-ids "$DATA_CENTER_IDS" \
  --container-disk-in-gb "${RUNPOD_CONTAINER_DISK_GB:-100}" \
  --network-volume-id "$VOLUME_ID" \
  --volume-mount-path /workspace \
  --ports "22/tcp,8787/http,9200/http" \
  --env "$ENV_JSON" \
  -o json)"
POD_ID="$(printf '%s' "$RESPONSE" | jq -r '.id')"
echo "$POD_ID" > "$STATE_DIR/pod-id"

echo "waiting for public SSH..." >&2
for _ in $(seq 1 96); do
  STATUS="$(runpodctl pod get "$POD_ID" -o json 2>/dev/null || true)"
  PARSED="$(printf '%s' "$STATUS" | python3 -c '
import json, sys
raw=sys.stdin.read().strip()
if not raw:
    print("WAIT")
    raise SystemExit
d=json.loads(raw)
runtime=d.get("runtime") or {}
ports=runtime.get("ports") or []
ssh=next((p for p in ports if int(p.get("privatePort",0)) == 22 and p.get("isIpPublic")), None)
if ssh and ssh.get("ip") and ssh.get("publicPort"):
    print(f"READY {ssh[\"ip\"]} {ssh[\"publicPort\"]}")
else:
    print("WAIT")
')"
  if [[ "$PARSED" == READY* ]]; then
    read -r _ IP PORT <<<"$PARSED"
    printf '%s %s\n' "$IP" "$PORT" > "$STATE_DIR/ssh"
    echo "pod ready: ssh -p $PORT root@$IP" >&2
    exit 0
  fi
  sleep 5
done

echo "pod created but SSH was not ready in time: $POD_ID" >&2
exit 1
