# OpenLens RunPod Serverless Encoder

This image runs the real OpenLens encoder path behind RunPod Serverless:

- `colpali` by default for multi-vector PDF/image patch embeddings.
- `qwen` is also installed for explicit Qwen multimodal runs.
- Workers scale to zero by default; creating the endpoint does not keep a GPU alive.

The handler accepts RunPod `input` payloads shaped like:

```json
{
  "backend": "colpali",
  "records": [{ "doc_id": "demo", "source": "fixture", "source_id": "demo", "source_url": "https://example.test", "modality": "document", "title": "Demo" }],
  "return_records": true
}
```

For larger jobs, pass `records_url` and `output_url` so the queue payload does not carry the whole corpus:

```json
{
  "backend": "colpali",
  "records_url": "https://signed.example/records.jsonl",
  "output_url": "https://signed.example/encoded.jsonl",
  "return_records": false
}
```
