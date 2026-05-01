# Verification: 10k Multimodal OpenLens Run

Date: May 1, 2026

Environment:

- OpenSearch 3.3.0 at `http://localhost:9200`
- Index: `openlens_multimodal`
- Vector dimension: `384`
- Embedding backend: `feature-hash`
- Qwen model setting: `qwen2b`

Commands run:

```bash
uv run openlens-build --bulk-internet-archive --target-docs 10000 --ia-page-size 1000
uv run openlens-index
uv run openlens-smoke --query "public domain video spacewalk" --mode lir --top-k 5
uv run openlens-smoke --query "archival audio speeches" --mode hybrid --top-k 5
```

Corpus build output:

```text
nasa: 500 SQL-style table records
ia-texts: 2,375 records
ia-images: 2,375 records
ia-videos: 2,375 records
ia-audio: 2,375 records
Saved 10,000 records -> data/processed/open_corpus.jsonl
```

Indexed output:

```text
Embedded 10,000 records -> data/processed/open_corpus_embedded.jsonl
Indexed 10,000 documents into openlens_multimodal
```

OpenSearch count:

```json
{"count":10000,"_shards":{"total":1,"successful":1,"skipped":0,"failed":0}}
```

Corpus distribution:

| modality | count |
|---|---:|
| audio | 2,375 |
| image | 2,375 |
| pdf | 2,375 |
| table | 500 |
| video | 2,375 |

Source distribution:

| source | count |
|---|---:|
| Internet Archive | 9,500 |
| NASA Exoplanet Archive | 500 |

Patch-count distribution in `open_corpus_embedded.jsonl`:

| patch_count | docs |
|---:|---:|
| 2 | 5,719 |
| 3 | 3,342 |
| 4 | 208 |
| 5 | 53 |
| 6 | 178 |
| 10 | 500 |

API verification:

```json
{
  "opensearch": {
    "available": true,
    "detail": "OpenSearch 3.3.0 at http://localhost:9200",
    "doc_count": 10000
  },
  "index": "openlens_multimodal",
  "vector_dim": 384,
  "embedding_backend": "feature-hash",
  "qwen_model": "qwen2b"
}
```

Browser verification:

- Opened `http://localhost:8787`.
- UI displayed `10000 docs on OpenSearch`.
- Hybrid tab returned mixed PDF, image, video, and audio results.
- LIR tab returned OpenSearch results and the detail pane showed patch metadata.
- Screenshot captured locally at `output/playwright/openlens-10k-lir.png` (ignored by Git).
