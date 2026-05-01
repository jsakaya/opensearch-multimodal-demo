# OpenSearch 3.6 ColPali Path

OpenLens now follows the current OpenSearch visual-document late-interaction
shape:

- OpenSearch 3.6 local stack: `opensearchproject/opensearch:3.6.0`.
- First-stage retrieval: HNSW k-NN over the mean-pooled `vector` field, plus
  BM25/SQL candidates.
- Late interaction: ColPali query multi-vectors rescoring document/page/image
  multi-vectors stored in the disabled-object `colbert_vectors` field.
- Local fallback: the deterministic feature-hash backend still mirrors the same
  schema by copying `patch_vectors` into `colbert_vectors`, so the app and tests
  exercise the OpenSearch LIR contract without requiring a GPU.

Official references used for the implementation:

- OpenSearch 3.6 release: https://opensearch.org/blog/introducing-opensearch-3-6/
- OpenSearch 3.6 artifacts: https://opensearch.org/artifacts/by-version/
- OpenSearch late interaction reranking:
  https://docs.opensearch.org/latest/search-plugins/search-relevance/rerank-by-field-late-interaction/
- OpenSearch multimodal benchmark with ColPali:
  https://opensearch.org/blog/benchmarking-multimodal-document-search-in-opensearch-three-approaches-compared/
- Hugging Face ColPali model docs:
  https://huggingface.co/docs/transformers/en/model_doc/colpali

## Run Locally

```bash
docker compose up -d
uv run openlens-index
uv run openlens-smoke --query "NASA technical reports about exoplanet occurrence rates" --mode lir --top-k 5
```

The local default remains `feature-hash` unless a GPU ColPali environment is
selected.

## Run On H100/H200

```bash
uv sync --extra colpali
OPENLENS_EMBEDDING_BACKEND=colpali \
OPENLENS_COLPALI_MODEL=colpali-v1.3 \
OPENLENS_VECTOR_DIM=128 \
OPENLENS_COLPALI_BATCH_SIZE=4 \
OPENLENS_COLPALI_MAX_PATCH_VECTORS=1024 \
uv run openlens-index
```

Tune batch size in the pod:

```bash
uv run openlens-colpali-benchmark --model colpali-v1.3 --dimension 128 --max-batch 16
```

## Important Caveat

ColPali is a visual-document retrieval model. It is the right path for rendered
PDF pages, screenshots, diagrams, tables, and image-heavy records. Audio remains
transcript/caption/metadata retrieval until an audio-native enrichment path is
enabled.

Local macOS development can stay on Python 3.13 for the feature-hash/OpenSearch
demo. The GPU ColPali/Qwen extras install Torch only on Python <3.13; the RunPod
image uses Ubuntu Python 3.10 with CUDA Torch wheels.
