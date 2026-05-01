.PHONY: help pod-up pod-down gpu-demo gpu-demo-small runpod run tune-colpali tune-qwen ssh

OPENLENS_TARGET_DOCS ?= 10000
OPENLENS_BENCHMARK_REPEATS ?= 2
OPENLENS_BENCHMARK_SAMPLES_PER_MODALITY ?= 5

help:
	@echo "OpenLens RunPod + OpenSearch demo loop:"
	@echo ""
	@echo "  make pod-up            Create an H200/H100 RunPod pod for OpenLens"
	@echo "  make gpu-demo          Build/index $(OPENLENS_TARGET_DOCS) docs on the pod and open a tunnel"
	@echo "  make gpu-demo-small    Fast smoke: 200 docs, 1 benchmark repeat"
	@echo "  make tune-colpali      Tune ColPali batch size on the running pod"
	@echo "  make tune-qwen         Tune Qwen batch size on the running pod"
	@echo "  make run CMD='...'     Run a command on the pod"
	@echo "  make ssh               Interactive SSH login"
	@echo "  make pod-down          Terminate the registered pod"
	@echo ""
	@echo "Useful env:"
	@echo "  RUNPOD_GPU_ID=\"NVIDIA H200\" or \"NVIDIA H100 SXM\""
	@echo "  RUNPOD_DATA_CENTER_IDS=US-CA-2"
	@echo "  RUNPOD_VOLUME_ID=t0ys2ffnll"
	@echo "  OPENLENS_TARGET_DOCS=10000"

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
	@OPENLENS_TARGET_DOCS=200 \
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
