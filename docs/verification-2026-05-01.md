# Verification 2026-05-01

Local OpenSearch 3.6.0 was rebuilt with the NASA customer-demo corpus after the
ColPali field/path upgrade. The committed benchmark below is still the local
feature-hash baseline; the H100/H200 path now supports `OPENLENS_EMBEDDING_BACKEND=colpali`.

## Corpus Build

```bash
uv run openlens-build \
  --target-docs 10000 \
  --query "artemis moon mars earth exoplanet hubble webb mission control"
```

Output:

```text
nasa-images: 3,500 records
nasa-videos: 1,800 records
nasa-audio: 94 records
ntrs: 2,400 records
nasa-exoplanets: 1,800 records
space demo shortfall: fetching 406 extra NASA images
space demo shortfall: fetching 406 extra exoplanet rows
Saved 10,000 records -> data/processed/open_corpus.jsonl
```

## Distribution

| Modality | Count |
|---|---:|
| image | 3,500 |
| video | 1,800 |
| audio | 94 |
| pdf | 2,400 |
| table | 2,206 |

| Source | Count |
|---|---:|
| NASA Image and Video Library | 5,394 |
| NASA STI Repository | 2,400 |
| NASA Exoplanet Archive | 2,206 |

Disallowed old-source metadata check: `0` records.

## Index And Smoke

```bash
uv run openlens-index
```

Output:

```text
Embedded 10,000 records -> data/processed/open_corpus_embedded.jsonl
Indexed 10,000 documents into openlens_multimodal
```

Smoke queries:

```bash
uv run openlens-smoke --query "Artemis moon landing mission video" --mode hybrid --top-k 5
uv run openlens-smoke --query "mission control audio schedule inventory" --mode hybrid --top-k 5
```

Observed top hits:

| Query | Top Hit | Modality |
|---|---|---|
| Artemis moon landing mission video | NASA's Artemis I Moon Mission: Launch to Splashdown Highlights | video |
| mission control audio schedule inventory | HWHAP Ep380 Mission Control: Schedule and Inventory | audio |

## Benchmark

```bash
uv run openlens-benchmark \
  --samples-per-modality 5 \
  --repeats 3 \
  --output docs/benchmarks/local-2026-05-01-nasa-feature-hash.json
```

Summary:

| Slice | n | p50 ms | p95 ms | exact@k | modality@1 | modality@3 |
|---|---:|---:|---:|---:|---:|---:|
| all | 366 | 41.12 | 197.52 | 96.00% | 87.70% | 95.90% |
| hybrid | 90 | 97.20 | 223.66 | 96.00% | 90.00% | 96.67% |
| keyword | 90 | 13.09 | 31.35 | 92.00% | 90.00% | 96.67% |
| lir | 90 | 72.19 | 199.67 | 96.00% | 83.33% | 93.33% |
| vector | 90 | 8.68 | 16.21 | 100.00% | 86.67% | 96.67% |
| sql | 6 | 16.65 | 318.17 | 0.00% | 100.00% | 100.00% |
