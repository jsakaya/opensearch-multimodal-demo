#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../_common.sh"

IMAGE="${IMAGE:-ghcr.io/jsakaya/openlens-encoder-serverless:latest}"
TEMPLATE_NAME="${TEMPLATE_NAME:-openlens-colpali-encoder-serverless}"
ENDPOINT_NAME="${ENDPOINT_NAME:-openlens-colpali-encoder}"
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-80}"
CONTAINER_REGISTRY_AUTH_ID="${CONTAINER_REGISTRY_AUTH_ID:-cmols3tco002dl707j9r6a08l}"
WORKERS_MIN="${WORKERS_MIN:-0}"
WORKERS_MAX="${WORKERS_MAX:-1}"
IDLE_TIMEOUT_S="${IDLE_TIMEOUT_S:-5}"
EXECUTION_TIMEOUT_MS="${EXECUTION_TIMEOUT_MS:-1800000}"
GPU_CANDIDATES="${GPU_CANDIDATES:-[\"NVIDIA L4\",\"NVIDIA A40\",\"NVIDIA RTX A5000\",\"NVIDIA RTX A6000\",\"NVIDIA GeForce RTX 4090\",\"NVIDIA L40S\"]}"

command -v jq >/dev/null || { echo "jq is required" >&2; exit 1; }
load_runpod_key

say() { printf "\033[1;36m==\033[0m %s\n" "$*"; }

extract_id_or_die() {
  local file="$1"
  local label="$2"
  local id
  id="$(jq -r 'if type=="object" then .id // empty else empty end' "$file")"
  if [[ -z "$id" ]]; then
    echo "${label} request failed:" >&2
    cat "$file" >&2
    exit 1
  fi
  printf '%s\n' "$id"
}

TEMPLATE_ENV="$(jq -n \
  --arg backend "${OPENLENS_EMBEDDING_BACKEND:-colpali}" \
  --arg colpali_model "${OPENLENS_COLPALI_MODEL:-colpali-v1.3}" \
  --arg vector_dim "${OPENLENS_VECTOR_DIM:-128}" \
  --arg colpali_batch "${OPENLENS_COLPALI_BATCH_SIZE:-2}" \
  --arg colpali_pages "${OPENLENS_COLPALI_MAX_PAGES:-1}" \
  --arg colpali_patch_vectors "${OPENLENS_COLPALI_MAX_PATCH_VECTORS:-1024}" \
  --arg max_records "${OPENLENS_SERVERLESS_MAX_RECORDS:-10000}" \
  '{
    OPENLENS_EMBEDDING_BACKEND: $backend,
    OPENLENS_COLPALI_MODEL: $colpali_model,
    OPENLENS_VECTOR_DIM: $vector_dim,
    OPENLENS_COLPALI_BATCH_SIZE: $colpali_batch,
    OPENLENS_COLPALI_MAX_PAGES: $colpali_pages,
    OPENLENS_COLPALI_MAX_PATCH_VECTORS: $colpali_patch_vectors,
    OPENLENS_SERVERLESS_MAX_RECORDS: $max_records,
    HF_HUB_DISABLE_XET: "1",
    HF_HOME: "/runpod-volume/.cache/huggingface",
    XDG_CACHE_HOME: "/runpod-volume/.cache",
    UV_CACHE_DIR: "/runpod-volume/.cache/uv"
  }')"

say "looking for existing template '${TEMPLATE_NAME}'"
TEMPLATE_ID="$(api GET /templates | jq -r --arg n "$TEMPLATE_NAME" '.[] | select(.name==$n) | .id' | head -1)"

if [[ -z "$TEMPLATE_ID" ]]; then
  say "creating zero-idle serverless template"
  jq -n \
    --arg n "$TEMPLATE_NAME" \
    --arg img "$IMAGE" \
    --arg registry_auth "$CONTAINER_REGISTRY_AUTH_ID" \
    --argjson disk "$CONTAINER_DISK_GB" \
    --argjson env "$TEMPLATE_ENV" \
    '{
      name: $n,
      imageName: $img,
      containerDiskInGb: $disk,
      containerRegistryAuthId: $registry_auth,
      isServerless: true,
      env: $env,
      ports: []
    }' > /tmp/openlens-serverless-template-body.json
  api POST /templates "$(cat /tmp/openlens-serverless-template-body.json)" > /tmp/openlens-serverless-template.json
  TEMPLATE_ID="$(extract_id_or_die /tmp/openlens-serverless-template.json template)"
else
  say "updating template '${TEMPLATE_NAME}'"
  jq -n \
    --arg n "$TEMPLATE_NAME" \
    --arg img "$IMAGE" \
    --arg registry_auth "$CONTAINER_REGISTRY_AUTH_ID" \
    --argjson env "$TEMPLATE_ENV" \
    '{
      name: $n,
      imageName: $img,
      containerRegistryAuthId: $registry_auth,
      env: $env
    }' > /tmp/openlens-serverless-template-update-body.json
  api PATCH "/templates/${TEMPLATE_ID}" "$(cat /tmp/openlens-serverless-template-update-body.json)" \
    > /tmp/openlens-serverless-template-update.json
  extract_id_or_die /tmp/openlens-serverless-template-update.json template-update >/dev/null
fi
echo "template id: ${TEMPLATE_ID}"

say "looking for existing endpoint '${ENDPOINT_NAME}'"
ENDPOINT_ID="$(api GET /endpoints | jq -r --arg n "$ENDPOINT_NAME" '.[] | select(.name==$n) | .id' | head -1)"

if [[ -z "$ENDPOINT_ID" ]]; then
  say "creating endpoint with workersMin=${WORKERS_MIN}"
  jq -n \
    --arg n "$ENDPOINT_NAME" \
    --arg tid "$TEMPLATE_ID" \
    --argjson gpus "$GPU_CANDIDATES" \
    --argjson workers_min "$WORKERS_MIN" \
    --argjson workers_max "$WORKERS_MAX" \
    --argjson idle_timeout "$IDLE_TIMEOUT_S" \
    --argjson timeout "$EXECUTION_TIMEOUT_MS" \
    '{
      name: $n,
      templateId: $tid,
      gpuTypeIds: $gpus,
      scalerType: "QUEUE_DELAY",
      scalerValue: 4,
      workersMin: $workers_min,
      workersMax: $workers_max,
      idleTimeout: $idle_timeout,
      executionTimeoutMs: $timeout,
      gpuCount: 1
    }' > /tmp/openlens-serverless-endpoint-body.json
  api POST /endpoints "$(cat /tmp/openlens-serverless-endpoint-body.json)" > /tmp/openlens-serverless-endpoint.json
  ENDPOINT_ID="$(extract_id_or_die /tmp/openlens-serverless-endpoint.json endpoint)"
else
  say "updating endpoint '${ENDPOINT_NAME}' with workersMin=${WORKERS_MIN}"
  jq -n \
    --arg n "$ENDPOINT_NAME" \
    --arg tid "$TEMPLATE_ID" \
    --argjson gpus "$GPU_CANDIDATES" \
    --argjson workers_min "$WORKERS_MIN" \
    --argjson workers_max "$WORKERS_MAX" \
    --argjson idle_timeout "$IDLE_TIMEOUT_S" \
    --argjson timeout "$EXECUTION_TIMEOUT_MS" \
    '{
      name: $n,
      templateId: $tid,
      gpuTypeIds: $gpus,
      scalerType: "QUEUE_DELAY",
      scalerValue: 4,
      workersMin: $workers_min,
      workersMax: $workers_max,
      idleTimeout: $idle_timeout,
      executionTimeoutMs: $timeout,
      gpuCount: 1
    }' > /tmp/openlens-serverless-endpoint-update-body.json
  api PATCH "/endpoints/${ENDPOINT_ID}" "$(cat /tmp/openlens-serverless-endpoint-update-body.json)" \
    > /tmp/openlens-serverless-endpoint-update.json
  extract_id_or_die /tmp/openlens-serverless-endpoint-update.json endpoint-update >/dev/null
fi

printf '%s\n' "$TEMPLATE_ID" > "${STATE_DIR}/serverless-template-id"
printf '%s\n' "$ENDPOINT_ID" > "${STATE_DIR}/serverless-endpoint-id"
echo "endpoint id: ${ENDPOINT_ID}"
printf "\033[1;32mEndpoint URL:\033[0m https://api.runpod.ai/v2/%s\n" "${ENDPOINT_ID}"
printf "\033[1;32mNo worker has been invoked by this script.\033[0m workersMin=%s idleTimeout=%ss\n" \
  "$WORKERS_MIN" "$IDLE_TIMEOUT_S"
