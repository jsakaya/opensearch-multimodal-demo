#!/usr/bin/env bash
# Create an H100 RunPod pod for OpenLens Qwen3-VL-Embedding encoding.
#
# Required:
#   RUNPOD_API_KEY
#   RUNPOD_VOLUME_ID
#
# Optional:
#   RUNPOD_POD_NAME=openlens-qwen-h100
#   RUNPOD_GPU="NVIDIA H100 80GB HBM3"
#   RUNPOD_IMAGE=ghcr.io/jsakaya/openlens-qwen-encoder:latest
#   RUNPOD_PUBKEY=~/.ssh/id_ed25519.pub

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

require_env RUNPOD_API_KEY
require_env RUNPOD_VOLUME_ID

NAME="${RUNPOD_POD_NAME:-openlens-qwen-h100}"
GPU="${RUNPOD_GPU:-NVIDIA H100 80GB HBM3}"
IMAGE="${RUNPOD_IMAGE:-ghcr.io/jsakaya/openlens-qwen-encoder:latest}"
PUBKEY_FILE="${RUNPOD_PUBKEY:-$HOME/.ssh/id_ed25519.pub}"
[[ -f "$PUBKEY_FILE" ]] || { echo "no SSH public key at $PUBKEY_FILE" >&2; exit 1; }
PUBKEY="$(cat "$PUBKEY_FILE")"

if [[ -f "$STATE_DIR/pod-id" ]]; then
  echo "pod already registered: $(cat "$STATE_DIR/pod-id")" >&2
  exit 1
fi

REQUEST=$(NAME="$NAME" GPU="$GPU" IMAGE="$IMAGE" VOL="$RUNPOD_VOLUME_ID" PUBKEY="$PUBKEY" python3 <<'PY'
import json, os
print(json.dumps({
    "name": os.environ["NAME"],
    "imageName": os.environ["IMAGE"],
    "cloudType": "SECURE",
    "computeType": "GPU",
    "gpuTypeIds": [os.environ["GPU"]],
    "gpuCount": 1,
    "gpuTypePriority": "availability",
    "dataCenterPriority": "availability",
    "allowedCudaVersions": ["13.0", "12.9", "12.8"],
    "containerDiskInGb": 80,
    "networkVolumeId": os.environ["VOL"],
    "volumeMountPath": "/workspace",
    "ports": ["22/tcp"],
    "supportPublicIp": True,
    "env": {
        "PUBLIC_KEY": os.environ["PUBKEY"],
        "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
        "OPENLENS_EMBEDDING_BACKEND": "qwen",
        "OPENLENS_QWEN_MODEL": "qwen8b",
        "OPENLENS_VECTOR_DIM": "4096",
        "OPENLENS_QWEN_BATCH_SIZE": os.environ.get("OPENLENS_QWEN_BATCH_SIZE", "16"),
        "OPENLENS_QWEN_MAX_FRAMES": os.environ.get("OPENLENS_QWEN_MAX_FRAMES", "64"),
        "OPENLENS_QWEN_FPS": os.environ.get("OPENLENS_QWEN_FPS", "1.0"),
        "OPENLENS_REQUIRE_OPENSEARCH": "1",
    },
}))
PY
)

echo "creating $NAME ($GPU, image=$IMAGE, volume=$RUNPOD_VOLUME_ID)..." >&2
RESPONSE="$(api POST /pods "$REQUEST")"
POD_ID="$(printf '%s' "$RESPONSE" | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')"
echo "$POD_ID" > "$STATE_DIR/pod-id"

echo "waiting for public SSH..." >&2
for _ in $(seq 1 96); do
  STATUS="$(api GET "/pods/$POD_ID")"
  PARSED="$(printf '%s' "$STATUS" | python3 -c '
import json, sys
d=json.load(sys.stdin)
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
