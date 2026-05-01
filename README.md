# OpenLens: Multimodal OpenSearch Demo

OpenLens is a local-first OpenSearch demo for searching open multimodal records:

- NASA Image and Video Library images, videos, audio clips, and thumbnails.
- NASA STI Repository / NTRS technical PDF records and public document metadata.
- NASA Exoplanet Archive TAP rows treated as SQL-style table documents.

The demo indexes every item into one OpenSearch 3.6 schema with BM25 fields, facets, a pooled HNSW vector, and per-document multi-vectors for late-interaction reranking. Retrieval runs as keyword, vector, RRF hybrid, SQL, or LIR patch rerank search. The code is designed so a tiny local corpus and a million-document corpus use the same ingestion/index contract.

## Why This Shape

OpenSearch can combine classic inverted-index retrieval with vector search. This repo keeps the retrieval path explicit:

1. Normalize each modality into a shared `OpenRecord`.
2. Compose searchable evidence from titles, captions, abstracts, transcripts/descriptions, table cells, tags, licenses, and asset metadata.
3. Split each record into modality-aware patches: PDF/text pages, image captions/assets, video/audio chunks with timestamps, table cells, and mixed evidence.
4. Store a pooled dense vector in `vector` for HNSW k-NN and keep `patch_vectors` plus OpenSearch-docs-aligned `colbert_vectors` for late-interaction scoring.
5. Use `refresh_interval=1s` and `refresh=wait_for`/`refresh=true` on small upserts for near real-time retrieval.
6. Use streaming bulk indexing with deterministic IDs for scalable rebuilds and incremental replays.

The default embedder is deterministic and local so the 10k demo can run anywhere. For visual-document retrieval, set `OPENLENS_EMBEDDING_BACKEND=colpali` and use `vidore/colpali-v1.3-hf`; OpenLens stores the ColPali page/image token vectors in `colbert_vectors` and mean-pools them into the first-stage HNSW vector. The Qwen3-VL path remains available with `OPENLENS_EMBEDDING_BACKEND=qwen`.

## Data Sources

- NASA Image and Video Library metadata is read through `https://images-api.nasa.gov/search`.
- NASA STI Repository / NTRS records are read through `https://ntrs.nasa.gov/api/citations/search`.
- NASA Exoplanet Archive rows are read through the TAP SQL endpoint in CSV format.

All generated data under `data/processed/` is ignored by Git.

## Setup

```bash
uv sync --extra dev
docker compose up -d
```

Build a small open corpus:

```bash
uv run openlens-build --limit-per-source 8
```

Build the 10,000-record NASA/space corpus with images, videos, audio, technical
PDF records, and SQL-style exoplanet rows:

```bash
uv run openlens-build \
  --target-docs 10000 \
  --query "artemis moon mars earth exoplanet hubble webb mission control"
```

Optionally fetch available NASA STI full-text snippets:

```bash
uv run openlens-build --limit-per-source 4 --fetch-pdf-text
```

Embed and index:

```bash
uv run openlens-index
```

For a compact local machine index, use 128-dimensional feature-hash vectors:

```bash
OPENLENS_VECTOR_DIM=128 uv run openlens-index
```

If OpenSearch is not running, this still writes `data/processed/open_corpus_embedded.jsonl`:

```bash
uv run openlens-index --skip-opensearch
```

Run a smoke query:

```bash
uv run openlens-smoke --query "satellite imagery climate change"
uv run openlens-smoke --query "mission control audio schedule inventory" --mode lir
```

Run a repeatable OpenSearch benchmark:

```bash
uv run openlens-benchmark --output docs/benchmarks/local-openlens.json
```

See `docs/retrieval-quality-and-display.md` for the H100/H200 reranking, audio
enrichment, and LLM evidence-display plan. The latest committed local hard-number
run is in `docs/benchmarks/local-2026-05-01-nasa-feature-hash.md`.

Serve the app:

```bash
uv run openlens-api
```

Open http://localhost:8787.

## API Surface

```bash
curl http://localhost:8787/api/status
curl -X POST http://localhost:8787/api/prewarm
curl 'http://localhost:8787/api/search?q=satellite%20imagery%20climate%20change&mode=hybrid&top_k=5'
curl 'http://localhost:8787/api/search?q=Artemis%20moon%20landing%20mission%20video&mode=lir&top_k=5'
```

Near real-time ingest:

```bash
curl -X POST http://localhost:8787/api/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "title": "Live smoke note on coral reef thermal stress",
    "body": "Satellite thermal anomaly maps linked to coral bleaching risk.",
    "modality": "document",
    "source": "Verification note",
    "tags": ["coral", "thermal"]
  }'
```

With OpenSearch available, the API writes the record to JSONL for replay, indexes it into `openlens_multimodal`, uses `refresh=wait_for`, clears the app retriever cache, and makes it searchable immediately.

## Scaling Notes

- Increase `number_of_shards` in `openlens.indexer.index_mapping()` for multi-node clusters.
- Keep deterministic `doc_id` values and replay source cursors into `helpers.streaming_bulk`.
- Bulk load with `refresh=false`, then call one manual refresh at the end of a backfill.
- For live writes, index individual docs or small batches with `refresh=wait_for`.
- Split very large PDFs/transcripts into smaller patches or passage records that share `source_id`.
- Add an ingest pipeline for production OCR, speech-to-text, PDF extraction, and neural sparse vectors.

## ColPali / RunPod Encoder

For the OpenSearch 3.6 ColPali late-interaction path on H100/H200:

```bash
uv sync --extra colpali
OPENLENS_EMBEDDING_BACKEND=colpali \
OPENLENS_COLPALI_MODEL=colpali-v1.3 \
OPENLENS_VECTOR_DIM=128 \
OPENLENS_COLPALI_BATCH_SIZE=4 \
OPENLENS_COLPALI_MAX_PATCH_VECTORS=1024 \
uv run openlens-index
```

Tune the GPU batch size:

```bash
uv run openlens-colpali-benchmark --model colpali-v1.3 --dimension 128 --max-batch 16
```

The optional RunPod image lives in `docker/runpod-openlens-qwen-encoder/` and mirrors the QuickInsights Qwen3.5 fast-path setup:

- CUDA 12.9 builder/runtime stages.
- Prebuilt `/opt/venv` copied into the runtime image.
- `PUBLIC_KEY` SSH bootstrap on port 22.
- Build-time verification for Qwen, ColPali, PDF rendering, and OpenLens.
- GHCR workflow: `.github/workflows/build-runpod-openlens-qwen-encoder.yml`.

See `docs/runpod-gpu-plan.md` for the current `runpodctl` GPU/volume choice.
On local macOS Python 3.13, the base demo works, but the Torch-backed ColPali
and Qwen extras are intended for Python 3.10-3.12 or the RunPod CUDA image.

Full-power RunPod demo path:

```bash
# RUNPOD_API_KEY is read from env or macOS Keychain item runpod-api-key.
# Defaults: H200 SXM in US-CA-2 with the josephsakaya-unsloth-h100 volume.
make pod-up
make gpu-demo
```

For a faster H200/H100 smoke before the full 10k run:

```bash
make pod-up
make gpu-demo-small
```

`make gpu-demo` SSHes into the GPU pod, starts a single-node OpenSearch
3.6 service if `OPENSEARCH_URL` is not already reachable, autotunes the ColPali
batch size, builds the 10k NASA/space corpus, indexes 128-dimensional pooled
vectors plus ColPali multi-vectors into OpenSearch, starts the API, calls
`POST /api/prewarm`, writes retrieval benchmark artifacts, and opens a local SSH
tunnel. The browser URL prints as `http://127.0.0.1:8787`.

Inside an already-running pod you can run the same remote sequence directly:

```bash
scripts/runpod/run.sh 'bash /opt/openlens/scripts/runpod/full-power-demo-remote.sh'
```

OpenLens search APIs require OpenSearch by default (`OPENLENS_REQUIRE_OPENSEARCH=1`).
SQL/table retrieval is also OpenSearch-native through `_plugins/_sql`:

```bash
uv run openlens-smoke \
  --mode sql \
  --query "SELECT modality, COUNT(*) AS n FROM openlens GROUP BY modality"
```

## Validation

```bash
uv run pytest
uv run ruff check .
```

Verified locally on May 1, 2026 with OpenSearch 3.6.0:

- `uv run openlens-build --limit-per-source 3 --output /tmp/openlens-nasa-small.jsonl` fetched 15 NASA records.
- `uv run openlens-build --target-docs 10000 --query "artemis moon mars earth exoplanet hubble webb mission control"` builds the customer-facing NASA corpus.
- `uv run openlens-index` embeds and indexes the records into `openlens_multimodal`.
- `uv run openlens-smoke --query "Artemis moon landing mission video" --mode hybrid --top-k 5` uses the OpenSearch hybrid path.
- `uv run openlens-smoke --query "mission control audio schedule inventory" --mode lir --top-k 5` uses the OpenSearch LIR path.
- Browser verification should show the LIR tab, patch trail, and the current OpenSearch document count.

See `docs/verification-2026-05-01.md` for the exact NASA run log and distribution.
