# Retrieval Quality And Display Plan

OpenLens should demo multimodal retrieval as an evidence workspace, not a chatbot
that hides the search process.

## Quality Stack

OpenSearch remains the base platform:

1. **Normalize** every source into `OpenRecord` documents with modality, source,
   assets, facets, table fields, and deterministic IDs.
2. **Patch** records into retrievable evidence units: visual captions/assets,
   PDF/text chunks, video timestamp chunks, audio transcript/caption chunks, and
   table cells.
3. **Index** pooled vectors, ColPali/OpenSearch `colbert_vectors`, BM25 fields,
   facets, and table fields into OpenSearch.
4. **Retrieve in parallel** with BM25, vector k-NN, SQL, and LIR patch scoring.
5. **Fuse candidates** with reciprocal rank fusion and preserve component scores
   so the UI can explain why a record surfaced.
6. **Rerank on H100/H200** with ColPali late interaction for visual documents,
   then optionally Qwen3-VL-Reranker-8B or another multimodal reranker over the
   top 50-200 OpenSearch candidates.
7. **Synthesize only after retrieval** with an LLM that receives compact, cited
   evidence patches and returns an answer plus uncertainty and missing-evidence
   notes.

The production-quality path should be:

```text
query
  -> query rewrite / modality intent
  -> OpenSearch BM25 + kNN + SQL + ColPali LIR candidate retrieval
  -> optional Qwen multimodal reranker on H100/H200
  -> evidence clustering
  -> grounded LLM synthesis with citations
  -> UI evidence board
```

## Audio Retrieval

ColPali is strongest for rendered document pages and screenshot-like images.
Qwen3-VL-Embedding is strong for text, images, screenshots, video, and mixed
vision/text inputs. Audio should not be represented as raw audio by default.
The robust audio path is:

1. Ingest catalog metadata, subjects, creators, duration, and asset URLs.
2. Add transcript/caption/description patches for every audio record.
3. Optionally enrich with Whisper, Qwen3.5-Omni/Qwen3-Omni captioning, or an
   audio-native embedding model such as CLAP/ImageBind.
4. Store the enriched text/audio evidence in OpenSearch and keep the source URL
   visible.

Current implementation:

- `audio_caption`
- `audio_transcript_or_description`
- `audio_asset_metadata`

These patch types make audio search work with BM25, vector, hybrid, and LIR
without pretending that Qwen-VL is raw-audio-native.

## Display Model

The best LLM-facing display is an evidence board with four layers:

1. **Answer strip**: one short grounded summary, confidence, and missing evidence.
2. **Modality lanes**: video, audio, images, PDFs, and tables grouped separately
   so a customer can see coverage at a glance.
3. **Evidence cards**: each card shows title, source, license, score components,
   and the exact best patch.
4. **Inspection pane**: source link, asset preview, patch timeline/page/cell list,
   raw facets/table fields, and provenance.

For video and audio, the patch list should behave like a timeline. For PDFs, it
should behave like page snippets. For SQL/table rows, it should behave like a
compact record inspector. The LLM should summarize across these groups, but the
UI should always keep the raw OpenSearch evidence visible.

## Demo Claims To Make

- OpenSearch is the operational retrieval substrate.
- ColPali on H100/H200 gives visual-document multi-vector retrieval over
  rendered pages, images, and thumbnails.
- LIR gives patch/page-level evidence instead of opaque document-level hits.
- SQL/table retrieval lives in the same search experience.
- Audio retrieval is handled responsibly through transcript/caption evidence and
  can be upgraded to audio-native enrichment.

## Demo Claims To Avoid Until Benchmarked

- Do not claim raw-audio Qwen-VL embedding unless using an Omni/audio-native
  model path.
- Do not claim every PDF page was visually embedded unless
  `OPENLENS_COLPALI_MAX_PAGES` and the indexed vector counts show that coverage.
- Do not claim million-document latency until an actual million-document run is
  measured.
- Do not claim best-in-class relevance without a labeled eval set and reranker
  ablation.
