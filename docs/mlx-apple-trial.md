# Apple MLX Trial

OpenLens has an optional MLX path for local Apple Silicon smoke tests. This is
not the CUDA RunPod path; it is a smaller local trial that proves Apple GPU
encoding can produce OpenLens-compatible vectors and records.

## Commands

Use Python 3.12 for the full Qwen3-VL smoke. The Qwen3-VL processor still
imports Torch/Torchvision for image preprocessing, while inference runs on MLX.

```bash
uv run --python 3.12 --extra mlx openlens-mlx-smoke
```

This writes:

- `data/processed/mlx_smoke_moon.png`
- `data/processed/mlx_smoke_embedded.jsonl`

For a text-only smoke that does not need the Qwen3-VL image processor:

```bash
uv run --python 3.12 --extra mlx openlens-mlx-smoke --skip-qwen-vl
```

To embed a small corpus with the MLX text model without touching OpenSearch:

```bash
OPENLENS_EMBEDDING_BACKEND=mlx \
uv run --python 3.12 --extra mlx openlens-index \
  --input data/processed/space_sample.jsonl \
  --skip-opensearch
```

## Backend Shape

- `OPENLENS_EMBEDDING_BACKEND=mlx` uses
  `mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ` by default and emits
  1024-dimensional common vectors.
- `OPENLENS_EMBEDDING_BACKEND=mlx-qwen-vl` uses
  `mlx-community/Qwen3-VL-Embedding-2B-6bit` by default and emits
  2048-dimensional image/text vectors.
- The OpenSearch API reports MLX import/device status through `/api/status`.

The current local MLX path is a trial backend, not a complete replacement for
the RunPod CUDA modality router. PDF visual late interaction through ColQwen /
ColPali-style MLX weights is the next piece to harden after the Qwen3-VL smoke.
