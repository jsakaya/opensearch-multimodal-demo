# RunPod OpenLens Qwen Encoder

This image follows the same shape as the QuickInsights Qwen3.5 RunPod image:

- CUDA 12.9 builder/runtime stages.
- A prebuilt `/opt/venv` copied into a slimmer runtime.
- `PUBLIC_KEY` SSH bootstrap on port 22.
- Hugging Face and uv caches under `/workspace/.cache`.
- A build-time verifier for CUDA Torch, Qwen-VL utilities, Transformers Qwen3-VL modules, and OpenLens.

The image is for GPU encoding jobs:

```bash
source /opt/activate-openlens.sh
cd /workspace/opensearch
OPENLENS_EMBEDDING_BACKEND=qwen \
OPENLENS_QWEN_MODEL=qwen8b \
OPENLENS_VECTOR_DIM=4096 \
OPENLENS_QWEN_BATCH_SIZE=16 \
OPENLENS_QWEN_MAX_FRAMES=64 \
openlens-index --skip-opensearch
```

The H100 target is `Qwen/Qwen3-VL-Embedding-8B` at the full 4096-dimensional
embedding size. Run `openlens-qwen-benchmark --model qwen8b --dimension 4096
--max-frames 64 --max-batch 64` in the pod to find the largest stable batch
size for the exact GPU and media mix. A Qwen3.5-compatible path can be used the
same way once the processor/model class is available in Transformers.

Published image target:

```text
ghcr.io/jsakaya/openlens-qwen-encoder:latest
```
