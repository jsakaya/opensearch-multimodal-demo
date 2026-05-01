# RunPod Serverless Encoding

OpenLens now has a zero-idle serverless encoder path for the expensive GPU step.
The local machine can build the corpus, run OpenSearch, and manage indexing; the
GPU only wakes when an explicit encoding job is submitted.

## Cost Guardrails

- `make serverless-deploy` creates or updates a RunPod Serverless template and endpoint with `workersMin=0`.
- The deploy command does not call `/run` or `/runsync`, so it should not start a worker.
- `make serverless-smoke` is the explicit command that starts a worker for a one-record smoke job.
- Defaults target 48 GB class GPUs first: `NVIDIA L40S`, `NVIDIA A40`, then `NVIDIA RTX A6000`.

## Commands

```bash
make serverless-deploy
```

This writes:

- `.runpod/serverless-template-id`
- `.runpod/serverless-endpoint-id`

Run a tiny smoke only when we actually want to spend for a cold start:

```bash
make serverless-smoke
```

For larger batches, submit records via `records_url` and write encoded JSONL to
`output_url`; that avoids moving 10,000 vector-heavy documents through the
RunPod queue response body.

## Payload

Small inline smoke:

```json
{
  "input": {
    "backend": "colpali",
    "records": ["...OpenRecord objects..."],
    "return_records": true
  }
}
```

Large URL-based job:

```json
{
  "input": {
    "backend": "colpali",
    "records_url": "https://signed.example/open_corpus.jsonl",
    "output_url": "https://signed.example/open_corpus_colpali.jsonl",
    "return_records": false
  }
}
```

The output records preserve OpenLens fields plus `vector`, `patch_vectors`, and
`colbert_vectors` for OpenSearch HNSW candidate retrieval and late-interaction
reranking.
