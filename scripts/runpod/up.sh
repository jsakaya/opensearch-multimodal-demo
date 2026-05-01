#!/usr/bin/env bash
# Create a RunPod pod for OpenLens ColPali/Qwen multimodal encoding.
#
# Required:
#   RUNPOD_API_KEY, or a macOS Keychain item named runpod-api-key
#
# Optional:
#   RUNPOD_VOLUME_ID=t0ys2ffnll
#   RUNPOD_VOLUME_NAME=josephsakaya-unsloth-h100
#   RUNPOD_POD_NAME=openlens-colpali-a40
#   RUNPOD_GPU_ID="NVIDIA A40"
#   RUNPOD_DATA_CENTER_IDS=CA-MTL-1
#   RUNPOD_TEMPLATE_ID=r97liuwvkd
#   RUNPOD_IMAGE=ghcr.io/jsakaya/openlens-qwen-encoder:latest
#   RUNPOD_PUBKEY=~/.ssh/id_ed25519.pub
#   RUNPOD_NO_NETWORK_VOLUME=1
#   RUNPOD_VOLUME_GB=100

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

load_runpod_key

NAME="${RUNPOD_POD_NAME:-openlens-colpali-a40}"
GPU="${RUNPOD_GPU_ID:-${RUNPOD_GPU:-NVIDIA A40}}"
DATA_CENTER_IDS="${RUNPOD_DATA_CENTER_IDS:-CA-MTL-1}"
TEMPLATE_ID="${RUNPOD_TEMPLATE_ID:-r97liuwvkd}"
IMAGE="${RUNPOD_IMAGE:-ghcr.io/jsakaya/openlens-qwen-encoder:latest}"
PUBKEY_FILE="${RUNPOD_PUBKEY:-$HOME/.ssh/id_ed25519.pub}"
[[ -f "$PUBKEY_FILE" ]] || { echo "no SSH public key at $PUBKEY_FILE" >&2; exit 1; }
PUBKEY="$(cat "$PUBKEY_FILE")"

VOLUME_ARGS=(--volume-mount-path /workspace)
if [[ "${RUNPOD_NO_NETWORK_VOLUME:-0}" == "1" ]]; then
  VOLUME_ARGS+=(--volume-in-gb "${RUNPOD_VOLUME_GB:-100}")
  VOLUME_DESC="ephemeral ${RUNPOD_VOLUME_GB:-100}GB"
else
  VOLUME_ID="${RUNPOD_VOLUME_ID:-}"
  if [[ -z "$VOLUME_ID" ]]; then
    VOLUME_NAME="${RUNPOD_VOLUME_NAME:-josephsakaya-unsloth-h100}"
    VOLUME_ID="$(runpodctl network-volume list -o json | jq -r --arg name "$VOLUME_NAME" '.[] | select(.name == $name) | .id' | head -1)"
  fi
  [[ -n "$VOLUME_ID" ]] || { echo "missing RUNPOD_VOLUME_ID and could not resolve RUNPOD_VOLUME_NAME" >&2; exit 1; }
  VOLUME_ARGS+=(--network-volume-id "$VOLUME_ID")
  VOLUME_DESC="$VOLUME_ID"
fi

if [[ -f "$STATE_DIR/pod-id" ]]; then
  echo "pod already registered: $(cat "$STATE_DIR/pod-id")" >&2
  exit 1
fi

ENV_JSON=$(PUBKEY="$PUBKEY" python3 <<'PY'
import json, os
backend = os.environ.get("OPENLENS_EMBEDDING_BACKEND", "colpali")
default_dim = "4096" if backend == "qwen" else "128"
print(json.dumps({
    "PUBLIC_KEY": os.environ["PUBKEY"],
    "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
    "OPENLENS_EMBEDDING_BACKEND": backend,
    "OPENLENS_QWEN_MODEL": "qwen8b",
    "OPENLENS_COLPALI_MODEL": os.environ.get("OPENLENS_COLPALI_MODEL", "colpali-v1.3"),
    "OPENLENS_VECTOR_DIM": os.environ.get("OPENLENS_VECTOR_DIM", default_dim),
    "OPENLENS_COLPALI_BATCH_SIZE": os.environ.get("OPENLENS_COLPALI_BATCH_SIZE", "4"),
    "OPENLENS_COLPALI_MAX_PAGES": os.environ.get("OPENLENS_COLPALI_MAX_PAGES", "1"),
    "OPENLENS_COLPALI_MAX_PATCH_VECTORS": os.environ.get("OPENLENS_COLPALI_MAX_PATCH_VECTORS", "1024"),
    "OPENLENS_QWEN_BATCH_SIZE": os.environ.get("OPENLENS_QWEN_BATCH_SIZE", "16"),
    "OPENLENS_QWEN_MAX_FRAMES": os.environ.get("OPENLENS_QWEN_MAX_FRAMES", "64"),
    "OPENLENS_QWEN_FPS": os.environ.get("OPENLENS_QWEN_FPS", "1.0"),
    "OPENLENS_REQUIRE_OPENSEARCH": "1",
}))
PY
)

CREATE_IMAGE_ARGS=(--image "$IMAGE")
IMAGE_DESC="image=$IMAGE"
if [[ -n "$TEMPLATE_ID" ]]; then
  CREATE_IMAGE_ARGS=(--template-id "$TEMPLATE_ID")
  IMAGE_DESC="template=$TEMPLATE_ID"
fi

echo "creating $NAME ($GPU, data centers=$DATA_CENTER_IDS, $IMAGE_DESC, volume=$VOLUME_DESC)..." >&2
RESPONSE="$(runpodctl pod create \
  --name "$NAME" \
  "${CREATE_IMAGE_ARGS[@]}" \
  --cloud-type SECURE \
  --gpu-id "$GPU" \
  --gpu-count 1 \
  --data-center-ids "$DATA_CENTER_IDS" \
  --container-disk-in-gb "${RUNPOD_CONTAINER_DISK_GB:-100}" \
  "${VOLUME_ARGS[@]}" \
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
    print("READY {} {}".format(ssh["ip"], ssh["publicPort"]))
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
