# OpenLens: Multimodal OpenSearch Demo

OpenLens is a local-first OpenSearch demo for searching open multimodal records:

- Wikimedia Commons image files and machine-readable license metadata.
- arXiv paper/PDF records, with optional first-page PDF text extraction.
- Internet Archive text/PDF-like records, images, videos, audio records, and thumbnails.
- NASA Exoplanet Archive TAP rows treated as SQL-style table documents.

The demo indexes every item into one OpenSearch schema with BM25 fields, facets, a pooled HNSW `knn_vector`, and per-document patch vectors for late-interaction reranking. Retrieval runs as keyword, vector, RRF hybrid, or LIR patch rerank search. The code is designed so a tiny local corpus and a million-document corpus use the same ingestion/index contract.

## Why This Shape

OpenSearch can combine classic inverted-index retrieval with vector search. This repo keeps the retrieval path explicit:

1. Normalize each modality into a shared `OpenRecord`.
2. Compose searchable evidence from titles, captions, abstracts, transcripts/descriptions, table cells, tags, licenses, and asset metadata.
3. Split each record into modality-aware patches: PDF/text pages, image captions/assets, video/audio chunks with timestamps, table cells, and mixed evidence.
4. Store a pooled dense vector in `vector` for HNSW k-NN and keep `patch_vectors` for late-interaction scoring.
5. Use `refresh_interval=1s` and `refresh=wait_for`/`refresh=true` on small upserts for near real-time retrieval.
6. Use streaming bulk indexing with deterministic IDs for scalable rebuilds and incremental replays.

The default embedder is deterministic and local so the 10k demo can run anywhere. For native multimodal retrieval, set `OPENLENS_EMBEDDING_BACKEND=qwen` and use the Qwen3-VL-Embedding provider; `OPENLENS_QWEN_MODEL` can also point at a Qwen3.5-compatible local/HF model path once its processor exposes the same interface.

## Data Sources

- Wikimedia Commons metadata is read through the MediaWiki Action API `imageinfo` module with `iiprop=extmetadata`.
- arXiv metadata is read from the public Atom API; each record carries the abstract page and PDF URL.
- Internet Archive records are read through `advancedsearch.php` and use archive item thumbnails.
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

Build the 10,000-record multimodal corpus:

```bash
uv run openlens-build --bulk-internet-archive --target-docs 10000 --ia-page-size 1000
```

Build a more customer-friendly NASA/space corpus with images, videos, audio,
PDF-like papers, and SQL-style exoplanet rows:

```bash
uv run openlens-build \
  --customer-demo-space \
  --target-docs 10000 \
  --query "artemis moon mars earth exoplanet"
```

Optionally extract first pages from arXiv PDFs:

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
uv run openlens-smoke --query "archival audio speeches" --mode lir
```

Run a repeatable OpenSearch benchmark:

```bash
uv run openlens-benchmark --output docs/benchmarks/local-openlens.json
```

See `docs/retrieval-quality-and-display.md` for the H100 reranking, audio
enrichment, and LLM evidence-display plan. The latest committed local hard-number
run is in `docs/benchmarks/local-2026-05-01-feature-hash.md`.

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
curl 'http://localhost:8787/api/search?q=public%20domain%20video%20spacewalk&mode=lir&top_k=5'
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

## Qwen / RunPod Encoder

For the best GPU-native multimodal encoding on H100:

```bash
uv sync --extra qwen
OPENLENS_EMBEDDING_BACKEND=qwen \
OPENLENS_QWEN_MODEL=qwen8b \
OPENLENS_VECTOR_DIM=4096 \
OPENLENS_QWEN_BATCH_SIZE=16 \
OPENLENS_QWEN_MAX_FRAMES=64 \
uv run openlens-index
```

The optional RunPod image lives in `docker/runpod-openlens-qwen-encoder/` and mirrors the QuickInsights Qwen3.5 fast-path setup:

- CUDA 12.9 builder/runtime stages.
- Prebuilt `/opt/venv` copied into the runtime image.
- `PUBLIC_KEY` SSH bootstrap on port 22.
- Build-time `verify_openlens_qwen.py`.
- GHCR workflow: `.github/workflows/build-runpod-openlens-qwen-encoder.yml`.

Full-power H100 demo path:

```bash
export RUNPOD_API_KEY=...
export RUNPOD_VOLUME_ID=...
scripts/runpod/up.sh
scripts/runpod/full-power-demo.sh
```

`full-power-demo.sh` SSHes into the H100 pod, starts a single-node OpenSearch
3.3 service if `OPENSEARCH_URL` is not already reachable, autotunes the Qwen
batch size up to `OPENLENS_QWEN_MAX_BATCH=64`, builds the 10k NASA/space corpus,
indexes full 4096-dimensional Qwen vectors and patch vectors into OpenSearch,
starts the API, calls `POST /api/prewarm`, writes retrieval benchmark artifacts,
and opens a local SSH tunnel. The browser URL prints as `http://127.0.0.1:8787`.

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

Verified locally on May 1, 2026 with OpenSearch 3.3.0:

- `uv run openlens-build --limit-per-source 5` fetched 19 public records.
- `uv run openlens-index` indexed 19 documents into `openlens_multimodal`.
- `uv run openlens-smoke --query "satellite imagery climate change" --top-k 5` used the OpenSearch hybrid path.
- `POST /api/ingest` inserted a live document and a follow-up search returned it as the top hit.
- `uv run openlens-build --bulk-internet-archive --target-docs 10000 --ia-page-size 1000` fetched exactly 10,000 public records: 2,375 each for IA text/PDF-like records, images, videos, audio records, plus 500 NASA table rows.
- `uv run openlens-index` embedded and indexed those 10,000 records into `openlens_multimodal`.
- `uv run openlens-smoke --query "public domain video spacewalk" --mode lir --top-k 5` used the OpenSearch LIR path.
- Browser verification showed the LIR tab, patch trail, and `10000 docs on OpenSearch`.

See `docs/verification-2026-05-01.md` for the exact 10k run log and distribution.
