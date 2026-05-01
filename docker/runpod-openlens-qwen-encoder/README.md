# RunPod OpenLens Multimodal Encoder

This image follows the same shape as the QuickInsights Qwen3.5 RunPod image:

- CUDA 12.9 builder/runtime stages.
- A prebuilt `/opt/venv` copied into a slimmer runtime.
- `PUBLIC_KEY` SSH bootstrap on port 22.
- Hugging Face and uv caches under `/workspace/.cache`.
- A build-time verifier for CUDA Torch, Qwen-VL, ColPali, PDF rendering, and OpenLens.

The image is for GPU encoding jobs:

```bash
source /opt/activate-openlens.sh
cd /workspace/opensearch
OPENLENS_EMBEDDING_BACKEND=colpali \
OPENLENS_COLPALI_MODEL=colpali-v1.3 \
OPENLENS_VECTOR_DIM=128 \
OPENLENS_COLPALI_BATCH_SIZE=4 \
OPENLENS_COLPALI_MAX_PATCH_VECTORS=1024 \
openlens-index --skip-opensearch
```

The H100/H200 ColPali target is `vidore/colpali-v1.3-hf`: 128-dimensional
multi-vectors stored as OpenSearch `colbert_vectors` plus mean-pooled HNSW
vectors. Run `openlens-colpali-benchmark --model colpali-v1.3 --dimension 128
--max-batch 16` in the pod to find the largest stable page batch size. The image
also still supports the Qwen path with `OPENLENS_EMBEDDING_BACKEND=qwen` and
`openlens-qwen-benchmark`.

Published image target:

```text
ghcr.io/jsakaya/openlens-qwen-encoder:latest
```
