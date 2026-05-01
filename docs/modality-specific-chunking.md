# Modality-Specific Chunking And Embedding

OpenLens now defaults to `OPENLENS_EMBEDDING_BACKEND=modality-router`. That keeps
one OpenSearch index while routing each record through a modality-specific chunk
and embedding plan.

## Index Contract

Every record still gets:

- `search_text`: BM25 / hybrid recall text.
- `vector`: 384-dimensional common recall vector for cross-corpus search.
- `patches`: inspectable evidence chunks for the UI.
- `patch_vectors`: portable local patch vectors for smoke tests.

Routed records also get one native first-stage vector field when applicable:

| Modality | Chunking | Native vector field | Dimension | Late-interaction field |
|---|---|---:|---:|---|
| `video` | 30s windows, 5s overlap, SentrySearch-style clip evidence | `qwen_vector` | 4096 | fallback patch evidence |
| `image` | caption plus asset patches | `qwen_vector` | 4096 | `colbert_vectors` |
| `pdf` | page/image evidence | `pdf_vector` | 128 | `colbert_vectors` |
| `audio` | 30s transcript/metadata windows plus raw audio source when enabled | `audio_vector` | 512 | fallback patch evidence |
| `table` | row and cell evidence | `table_vector` | 384 | SQL plugin path |
| `document` | text semantic chunks | `text_vector` | 384 | fallback patch evidence |

The OpenSearch mapping defines all vector fields, but `opensearch_source()` omits
empty modality vectors before indexing. That matters because OpenSearch rejects
empty arrays for `knn_vector` fields.

## Runtime Modes

CPU-safe local mode is the default:

```bash
uv run openlens-index --skip-opensearch
```

It emits deterministic vectors with the same dimensions as the production fields,
so schema, retrieval, UI, and JSONL handling can be tested without starting a GPU.

GPU mode activates the real encoders:

```bash
OPENLENS_USE_REAL_MODALITY_ENCODERS=1 \
OPENLENS_EMBEDDING_BACKEND=modality-router \
uv run openlens-index
```

The router uses:

- Qwen3-VL embedding for image/video vectors.
- ColPali-compatible page/patch vectors for PDFs and visual reranking.
- CLAP text/audio space for audio-native retrieval when
  `uv sync --extra audio` is installed.
- Row/cell dense text vectors plus OpenSearch SQL for tables.

## Retrieval

For a modality filter, vector search targets the native field:

- `modality=video` or `image` -> `qwen_vector`
- `modality=audio` -> `audio_vector`
- `modality=pdf` -> `pdf_vector`
- `modality=table` -> `table_vector`

Without a modality filter, OpenLens searches the common `vector` field and fuses
it with BM25. LIR mode uses OpenSearch candidate retrieval, then reranks with
`lateInteractionScore` over `colbert_vectors` where present.
