# RunPod GPU Plan

Checked with `runpodctl` on 2026-05-01 using the local Keychain credential
`runpod-api-key`.

## Credential Path

- `runpodctl` works when `RUNPOD_API_KEY` is loaded from macOS Keychain.
- `scripts/runpod/up.sh` now loads that key automatically if the environment
  variable is not already set.
- Secrets are not stored in this repo.

## Existing RunPod State

- Existing network volume: `t0ys2ffnll`
- Volume name: `josephsakaya-unsloth-h100`
- Data center: `US-CA-2`
- Size: `100 GB`

## GPU Choice

Recommended default for this demo: `NVIDIA H200` in `US-CA-2`.

Why:

- It keeps the pod near the existing `US-CA-2` network volume.
- It gives substantially more VRAM than H100 SXM for ColPali page/image
  multi-vector encoding, Qwen embedding, and reranking batch size tuning.
- `runpodctl datacenter list` showed `H200 SXM` available in `US-CA-2` while
  several H100/B200 options were low stock.
- It is safer than jumping to B200 for this image because the current container
  path is already validated on CUDA 12.9 / PyTorch / Qwen/ColPali modules and
  H100/H200 is the least surprising production demo target.

Override knobs:

```bash
RUNPOD_GPU_ID="NVIDIA H200" \
RUNPOD_DATA_CENTER_IDS=US-CA-2 \
RUNPOD_VOLUME_ID=t0ys2ffnll \
scripts/runpod/up.sh
```

For a maximum-flex benchmark run, try B200 later with:

```bash
RUNPOD_GPU_ID="NVIDIA B200" \
RUNPOD_DATA_CENTER_IDS=US-CA-2 \
scripts/runpod/up.sh
```

Do that only after the H200 path is clean, because the B200 stock was low and the
demo does not need 180 GB VRAM to show ColPali late interaction strongly.
