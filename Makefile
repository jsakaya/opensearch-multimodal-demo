.PHONY: help prep-cpu pod-up pod-down gpu-demo gpu-demo-small runpod run tune-colpali tune-qwen ssh serverless-deploy serverless-smoke

OPENLENS_TARGET_DOCS ?= 10000
OPENLENS_BENCHMARK_REPEATS ?= 2
OPENLENS_BENCHMARK_SAMPLES_PER_MODALITY ?= 5
OPENLENS_SPACE_QUERY ?= artemis moon mars earth exoplanet hubble webb mission control

help:
	@echo "OpenLens CPU-first + minimal-GPU demo loop:"
	@echo ""
	@echo "  make prep-cpu          Build $(OPENLENS_TARGET_DOCS) docs and verify local OpenSearch"
	@echo "  make serverless-deploy Create/update zero-idle RunPod Serverless encoder"
	@echo "  make serverless-smoke  Explicit one-record serverless encode smoke"
	@echo "  make pod-up            Create a smaller RunPod GPU pod only when needed"
	@echo "  make gpu-demo-small    GPU smoke: 25 docs, batch=1, no autotune"
	@echo "  make gpu-demo          GPU encode/index $(OPENLENS_TARGET_DOCS) docs and open a tunnel"
	@echo "  make tune-colpali      Tune ColPali batch size on the running pod"
	@echo "  make tune-qwen         Tune Qwen batch size on the running pod"
	@echo "  make run CMD='...'     Run a command on the pod"
	@echo "  make ssh               Interactive SSH login"
	@echo "  make pod-down          Terminate the registered pod"
	@echo ""
	@echo "Useful env:"
	@echo "  RUNPOD_GPU_ID=\"NVIDIA A40\" or \"NVIDIA L40S\"; use H100/H200 only deliberately"
	@echo "  RUNPOD_DATA_CENTER_IDS=CA-MTL-1"
	@echo "  RUNPOD_VOLUME_ID=t0ys2ffnll"
	@echo "  OPENLENS_TARGET_DOCS=10000"

prep-cpu:
	@docker compose up -d opensearch
	@uv run openlens-build \
	  --target-docs "$(OPENLENS_TARGET_DOCS)" \
	  --query "$(OPENLENS_SPACE_QUERY)" \
	  --output data/processed/open_corpus.jsonl
	@uv run openlens-index
	@uv run openlens-smoke --query "mission control audio schedule inventory" --mode lir --top-k 3
	@uv run openlens-smoke --query "NASA technical reports about exoplanet occurrence rates" --mode lir --top-k 3

serverless-deploy:
	@bash scripts/runpod/serverless/deploy_encoder.sh

serverless-smoke:
	@uv run python scripts/runpod/serverless/smoke_encoder.py --records 1

pod-up:
	@bash scripts/runpod/up.sh

pod-down:
	@bash scripts/runpod/down.sh

gpu-demo:
	@OPENLENS_TARGET_DOCS="$(OPENLENS_TARGET_DOCS)" \
	 OPENLENS_BENCHMARK_REPEATS="$(OPENLENS_BENCHMARK_REPEATS)" \
	 OPENLENS_BENCHMARK_SAMPLES_PER_MODALITY="$(OPENLENS_BENCHMARK_SAMPLES_PER_MODALITY)" \
	 bash scripts/runpod/full-power-demo.sh

gpu-demo-small:
	@OPENLENS_TARGET_DOCS=25 \
	 OPENLENS_COLPALI_BATCH_SIZE=1 \
	 OPENLENS_COLPALI_MAX_BATCH=2 \
	 OPENLENS_AUTOTUNE_COLPALI=0 \
	 OPENLENS_BENCHMARK_REPEATS=1 \
	 OPENLENS_BENCHMARK_SAMPLES_PER_MODALITY=2 \
	 bash scripts/runpod/full-power-demo.sh

runpod: gpu-demo

tune-colpali:
	@bash scripts/runpod/tune-colpali.sh

tune-qwen:
	@bash scripts/runpod/tune-qwen.sh

run:
	@test -n "$(CMD)" || (echo "usage: make run CMD='nvidia-smi'" >&2; exit 1)
	@bash scripts/runpod/run.sh '$(CMD)'

ssh:
	@bash scripts/runpod/run.sh
