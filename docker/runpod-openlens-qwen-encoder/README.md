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
OPENLENS_QWEN_MODEL=qwen2b \
OPENLENS_VECTOR_DIM=768 \
openlens-index --skip-opensearch
```

For a larger Qwen model, set `OPENLENS_QWEN_MODEL=qwen8b` or a local/HF model path
that exposes the Qwen3-VL embedding interface. A Qwen3.5-compatible path can be
used the same way once the processor/model class is available in Transformers.

Published image target:

```text
ghcr.io/jsakaya/openlens-qwen-encoder:latest
```
