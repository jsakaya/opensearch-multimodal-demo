from __future__ import annotations

import importlib.metadata as metadata
import importlib.util
import os

import torch


REQUIRED_DISTS = (
    "openlens-opensearch-demo",
    "transformers",
    "qwen-vl-utils",
    "torch",
    "accelerate",
)

REQUIRED_MODULES = (
    "openlens.qwen_embedder",
    "qwen_vl_utils",
    "transformers.models.qwen3_vl",
)


def version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "missing"


print(f"torch: {torch.__version__}")
print(f"cuda: {torch.version.cuda}")
print(f"cuda_available: {torch.cuda.is_available()}")

missing_dists = []
for dist in REQUIRED_DISTS:
    dist_version = version(dist)
    print(f"{dist}: {dist_version}")
    if dist_version == "missing":
        missing_dists.append(dist)

missing_modules = []
for module in REQUIRED_MODULES:
    available = importlib.util.find_spec(module) is not None
    print(f"{module}: {available}")
    if not available:
        missing_modules.append(module)

if torch.version.cuda is None:
    raise SystemExit("CUDA Torch wheel is missing")
if missing_dists:
    raise SystemExit(f"missing distributions: {', '.join(missing_dists)}")
if missing_modules:
    raise SystemExit(f"missing import modules: {', '.join(missing_modules)}")

if not torch.cuda.is_available():
    if os.environ.get("OPENLENS_QWEN_VERIFY_ALLOW_NO_GPU") == "1":
        print("openlens_qwen_encoder: deferred until a CUDA device is visible")
        raise SystemExit(0)
    raise SystemExit("CUDA device is unavailable")

print(f"gpu_name: {torch.cuda.get_device_name(0)}")
print("openlens_qwen_encoder: ready")
