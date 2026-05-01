# OpenLens: Multimodal OpenSearch Demo

OpenLens is a local-first OpenSearch demo for searching open multimodal records:

- Wikimedia Commons image files and machine-readable license metadata.
- arXiv paper/PDF records, with optional first-page PDF text extraction.
- Internet Archive video records and thumbnails.
- NASA Exoplanet Archive TAP rows treated as SQL-style table documents.

The demo indexes every item into one OpenSearch schema with BM25 fields, facets, a deterministic dense vector, and an HNSW `knn_vector`. Retrieval runs as keyword, vector, or application-side RRF hybrid search. The code is designed so a tiny local corpus and a million-document corpus use the same ingestion/index contract.

## Why This Shape

OpenSearch can combine classic inverted-index retrieval with vector search. This repo keeps the retrieval path explicit:

1. Normalize each modality into a shared `OpenRecord`.
2. Compose searchable evidence from titles, captions, abstracts, transcripts/descriptions, table cells, tags, licenses, and asset metadata.
3. Store a dense vector in `vector` for HNSW k-NN and keep rich raw fields for previews/facets.
4. Use `refresh_interval=1s` and `refresh=wait_for`/`refresh=true` on small upserts for near real-time retrieval.
5. Use streaming bulk indexing with deterministic IDs for scalable rebuilds and incremental replays.

For a production-scale version, swap the local deterministic `FeatureHashEmbedder` for OpenSearch ML Commons, a CLIP/text-image model, a hosted embedding API, or neural sparse search while keeping the index and UI contract.

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

Optionally extract first pages from arXiv PDFs:

```bash
uv run openlens-build --limit-per-source 4 --fetch-pdf-text
```

Embed and index:

```bash
uv run openlens-index
```

If OpenSearch is not running, this still writes `data/processed/open_corpus_embedded.jsonl`:

```bash
uv run openlens-index --skip-opensearch
```

Run a smoke query:

```bash
uv run openlens-smoke --query "satellite imagery climate change"
```

Serve the app:

```bash
uv run openlens-api
```

Open http://localhost:8787.

## API Surface

```bash
curl http://localhost:8787/api/status
curl 'http://localhost:8787/api/search?q=satellite%20imagery%20climate%20change&mode=hybrid&top_k=5'
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
- Split very large PDFs/transcripts into passage records that share `source_id`.
- Add an ingest pipeline for production OCR, speech-to-text, PDF extraction, and neural sparse vectors.

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
