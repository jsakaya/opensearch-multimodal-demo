#!/usr/bin/env bash
set -euo pipefail

source /opt/activate-openlens.sh

export OPENLENS_DATA_DIR="${OPENLENS_DATA_DIR:-/workspace/openlens-data}"
export OPENLENS_DOCS="${OPENLENS_DOCS:-$OPENLENS_DATA_DIR/open_corpus.jsonl}"
export OPENLENS_EMBEDDED_DOCS="${OPENLENS_EMBEDDED_DOCS:-$OPENLENS_DATA_DIR/open_corpus_embedded.jsonl}"
export OPENSEARCH_URL="${OPENSEARCH_URL:-http://127.0.0.1:9200}"
export OPENSEARCH_INDEX="${OPENSEARCH_INDEX:-openlens_multimodal}"
export OPENLENS_EMBEDDING_BACKEND="${OPENLENS_EMBEDDING_BACKEND:-colpali}"
export OPENLENS_QWEN_MODEL="${OPENLENS_QWEN_MODEL:-qwen8b}"
export OPENLENS_COLPALI_MODEL="${OPENLENS_COLPALI_MODEL:-colpali-v1.3}"
if [[ -z "${OPENLENS_VECTOR_DIM:-}" ]]; then
  if [[ "$OPENLENS_EMBEDDING_BACKEND" == "qwen" ]]; then
    export OPENLENS_VECTOR_DIM=4096
  else
    export OPENLENS_VECTOR_DIM=128
  fi
fi
export OPENLENS_COLPALI_BATCH_SIZE="${OPENLENS_COLPALI_BATCH_SIZE:-4}"
export OPENLENS_COLPALI_MAX_PAGES="${OPENLENS_COLPALI_MAX_PAGES:-1}"
export OPENLENS_COLPALI_MAX_PATCH_VECTORS="${OPENLENS_COLPALI_MAX_PATCH_VECTORS:-1024}"
export OPENLENS_QWEN_MAX_FRAMES="${OPENLENS_QWEN_MAX_FRAMES:-64}"
export OPENLENS_QWEN_FPS="${OPENLENS_QWEN_FPS:-1.0}"
export OPENLENS_REQUIRE_OPENSEARCH=1

mkdir -p "$OPENLENS_DATA_DIR"

ensure_opensearch() {
  if curl -fsS "$OPENSEARCH_URL" >/dev/null 2>&1; then
    return
  fi

  local version="${OPENLENS_OPENSEARCH_VERSION:-3.6.0}"
  local os_home="${OPENLENS_OPENSEARCH_HOME:-/workspace/opensearch-$version}"
  local tarball="${OPENLENS_OPENSEARCH_TARBALL:-https://artifacts.opensearch.org/releases/bundle/opensearch/$version/opensearch-$version-linux-x64.tar.gz}"

  if [[ ! -x "$os_home/bin/opensearch" ]]; then
    echo "downloading OpenSearch $version..."
    curl -fL "$tarball" -o /tmp/opensearch.tgz
    tar -xzf /tmp/opensearch.tgz -C /workspace
  fi

  mkdir -p "$OPENLENS_DATA_DIR/os-data" "$OPENLENS_DATA_DIR/os-logs" /workspace/openlens-os-home
  if ! id openlens-os >/dev/null 2>&1; then
    useradd -m -d /workspace/openlens-os-home -s /bin/bash openlens-os
  fi
  cat >"$os_home/config/opensearch.yml" <<EOF
cluster.name: openlens-gpu
node.name: openlens-gpu-1
discovery.type: single-node
network.host: 127.0.0.1
http.port: 9200
plugins.security.disabled: true
path.data: $OPENLENS_DATA_DIR/os-data
path.logs: $OPENLENS_DATA_DIR/os-logs
node.store.allow_mmap: false
EOF
  chown -R openlens-os:openlens-os "$os_home" "$OPENLENS_DATA_DIR/os-data" "$OPENLENS_DATA_DIR/os-logs" /workspace/openlens-os-home

  echo "starting OpenSearch on $OPENSEARCH_URL..."
  su -s /bin/bash openlens-os -c "OPENSEARCH_JAVA_OPTS='-Xms4g -Xmx4g' $os_home/bin/opensearch" \
    >"$OPENLENS_DATA_DIR/opensearch.log" 2>&1 &

  for _ in $(seq 1 90); do
    if curl -fsS "$OPENSEARCH_URL" >/dev/null 2>&1; then
      return
    fi
    sleep 2
  done
  echo "OpenSearch did not become ready. Last log lines:" >&2
  tail -80 "$OPENLENS_DATA_DIR/opensearch.log" >&2 || true
  exit 1
}

autotune_embedding() {
  if [[ "$OPENLENS_EMBEDDING_BACKEND" == "colpali" ]]; then
    if [[ "${OPENLENS_AUTOTUNE_COLPALI:-1}" != "1" ]]; then
      return
    fi
    local log="$OPENLENS_DATA_DIR/colpali-benchmark.log"
    echo "autotuning ColPali batch size on the GPU..."
    openlens-colpali-benchmark \
      --model "$OPENLENS_COLPALI_MODEL" \
      --dimension "$OPENLENS_VECTOR_DIM" \
      --max-patch-vectors "$OPENLENS_COLPALI_MAX_PATCH_VECTORS" \
      --max-batch "${OPENLENS_COLPALI_MAX_BATCH:-16}" | tee "$log"
    local tuned
    tuned="$(awk -F= '/OPENLENS_COLPALI_BATCH_SIZE=/{print $2}' "$log" | tail -1)"
    export OPENLENS_COLPALI_BATCH_SIZE="${tuned:-$OPENLENS_COLPALI_BATCH_SIZE}"
    echo "using OPENLENS_COLPALI_BATCH_SIZE=$OPENLENS_COLPALI_BATCH_SIZE"
    return
  fi

  if [[ "${OPENLENS_AUTOTUNE_QWEN:-1}" != "1" ]]; then
    export OPENLENS_QWEN_BATCH_SIZE="${OPENLENS_QWEN_BATCH_SIZE:-16}"
    return
  fi
  local log="$OPENLENS_DATA_DIR/qwen-benchmark.log"
  echo "autotuning Qwen batch size on the GPU..."
  openlens-qwen-benchmark \
    --model "$OPENLENS_QWEN_MODEL" \
    --dimension "$OPENLENS_VECTOR_DIM" \
    --max-frames "$OPENLENS_QWEN_MAX_FRAMES" \
    --max-batch "${OPENLENS_QWEN_MAX_BATCH:-64}" | tee "$log"
  local tuned
  tuned="$(awk -F= '/OPENLENS_QWEN_BATCH_SIZE=/{print $2}' "$log" | tail -1)"
  export OPENLENS_QWEN_BATCH_SIZE="${tuned:-${OPENLENS_QWEN_BATCH_SIZE:-16}}"
  echo "using OPENLENS_QWEN_BATCH_SIZE=$OPENLENS_QWEN_BATCH_SIZE"
}

build_corpus() {
  if [[ "${OPENLENS_REBUILD_CORPUS:-0}" == "1" || ! -s "$OPENLENS_DOCS" ]]; then
    echo "building customer demo corpus..."
    openlens-build \
      --target-docs "${OPENLENS_TARGET_DOCS:-10000}" \
      --query "${OPENLENS_SPACE_QUERY:-artemis moon mars earth exoplanet hubble webb mission control}" \
      --output "$OPENLENS_DOCS"
  fi
}

start_api() {
  pkill -f "uvicorn openlens.api:app" >/dev/null 2>&1 || true
  nohup uvicorn openlens.api:app --host 0.0.0.0 --port 8787 \
    >"$OPENLENS_DATA_DIR/openlens-api.log" 2>&1 &
  for _ in $(seq 1 60); do
    if curl -fsS http://127.0.0.1:8787/api/status >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  curl -fsS -X POST http://127.0.0.1:8787/api/prewarm | python -m json.tool
}

ensure_opensearch
autotune_embedding
build_corpus
echo "embedding and indexing $OPENLENS_EMBEDDING_BACKEND vectors into OpenSearch..."
openlens-index --input "$OPENLENS_DOCS"
start_api
openlens-benchmark \
  --samples-per-modality "${OPENLENS_BENCHMARK_SAMPLES_PER_MODALITY:-5}" \
  --repeats "${OPENLENS_BENCHMARK_REPEATS:-2}" \
  --output "$OPENLENS_DATA_DIR/retrieval-benchmark.json"

echo
echo "OpenLens full-power GPU demo is ready inside the pod."
echo "API: http://127.0.0.1:8787"
echo "OpenSearch: $OPENSEARCH_URL"
echo "Benchmark: $OPENLENS_DATA_DIR/retrieval-benchmark.md"
