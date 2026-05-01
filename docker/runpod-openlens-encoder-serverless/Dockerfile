# check=skip=InvalidBaseImagePlatform
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG TORCH_SPEC="torch>=2.11.0"
ARG TORCH_BACKEND="cu129"

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    HF_HOME=/runpod-volume/.cache/huggingface \
    XDG_CACHE_HOME=/runpod-volume/.cache \
    UV_CACHE_DIR=/runpod-volume/.cache/uv \
    HF_HUB_DISABLE_XET=1 \
    OPENLENS_EMBEDDING_BACKEND=colpali \
    OPENLENS_COLPALI_MODEL=colpali-v1.3 \
    OPENLENS_VECTOR_DIM=128 \
    OPENLENS_COLPALI_BATCH_SIZE=2 \
    OPENLENS_COLPALI_MAX_PAGES=1 \
    OPENLENS_COLPALI_MAX_PATCH_VECTORS=1024 \
    OPENLENS_SERVERLESS_INLINE_MAX_RECORDS=16 \
    OPENLENS_SERVERLESS_MAX_RECORDS=10000

RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
      ca-certificates \
      curl \
      ffmpeg \
      git \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      python3 \
      python3-pip \
      python3-venv && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "$VIRTUAL_ENV" && \
    python -m pip install --no-cache-dir --upgrade pip uv setuptools wheel packaging && \
    uv pip install --no-cache --upgrade --torch-backend="${TORCH_BACKEND}" \
      "${TORCH_SPEC}" \
      torchvision \
      torchaudio

COPY . /opt/openlens
WORKDIR /opt/openlens

RUN uv pip install --no-cache --upgrade -e "/opt/openlens[qwen,colpali]" "runpod>=1.7.9" && \
    python -c "from openlens.serverless_encoder import encode_payload; status = encode_payload({'action': 'status'}); assert status['ok'] is True; print(status['backend_default'])"

CMD ["python", "-u", "/opt/openlens/docker/runpod-openlens-encoder-serverless/handler.py"]
