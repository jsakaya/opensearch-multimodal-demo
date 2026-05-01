# check=skip=InvalidBaseImagePlatform
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu22.04 AS builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG TORCH_SPEC="torch>=2.11.0"
ARG TORCH_BACKEND="cu129"

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    HF_HOME=/workspace/.cache/huggingface \
    XDG_CACHE_HOME=/workspace/.cache \
    UV_CACHE_DIR=/workspace/.cache/uv

RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      ffmpeg \
      git \
      libcurl4-openssl-dev \
      libgl1 \
      libglib2.0-0 \
      pkg-config \
      python3-dev \
      python3-pip \
      python3-venv \
      rsync && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "$VIRTUAL_ENV"
RUN python -m pip install --upgrade pip uv setuptools wheel ninja packaging
RUN uv pip install --no-cache --upgrade --torch-backend="${TORCH_BACKEND}" \
      "${TORCH_SPEC}" \
      torchvision \
      torchaudio

COPY . /opt/openlens
RUN uv pip install --no-cache --upgrade -e "/opt/openlens[qwen]"

COPY docker/runpod-openlens-qwen-encoder/verify_openlens_qwen.py /opt/openlens/verify_openlens_qwen.py
RUN OPENLENS_QWEN_VERIFY_ALLOW_NO_GPU=1 python /opt/openlens/verify_openlens_qwen.py

RUN cat >/opt/activate-openlens.sh <<'EOF' && chmod +x /opt/activate-openlens.sh
#!/usr/bin/env bash
export VIRTUAL_ENV=/opt/venv
export PATH=/opt/venv/bin:$PATH
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
source /opt/venv/bin/activate
EOF

FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu22.04 AS runtime

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/venv \
    CUDA_HOME=/usr/local/cuda \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    HF_HOME=/workspace/.cache/huggingface \
    XDG_CACHE_HOME=/workspace/.cache \
    UV_CACHE_DIR=/workspace/.cache/uv \
    OPENLENS_EMBEDDING_BACKEND=qwen \
    OPENLENS_QWEN_MODEL=qwen8b \
    OPENLENS_VECTOR_DIM=4096 \
    OPENLENS_QWEN_BATCH_SIZE=16 \
    OPENLENS_QWEN_MAX_FRAMES=64 \
    OPENLENS_QWEN_FPS=1.0 \
    OPENLENS_REQUIRE_OPENSEARCH=1

RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
      ca-certificates \
      curl \
      ffmpeg \
      git \
      libgl1 \
      libglib2.0-0 \
      libgomp1 \
      openssh-server \
      python3 \
      python3-venv \
      rsync && \
    mkdir -p /run/sshd && \
    sed -ri 's/^#?PermitRootLogin .*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config && \
    sed -ri 's/^#?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/openlens /opt/openlens
COPY --from=builder /opt/activate-openlens.sh /opt/activate-openlens.sh
COPY docker/runpod-openlens-qwen-encoder/runpod-start.sh /opt/openlens/runpod-start.sh

RUN chmod +x /opt/openlens/runpod-start.sh && \
    OPENLENS_QWEN_VERIFY_ALLOW_NO_GPU=1 python /opt/openlens/verify_openlens_qwen.py

WORKDIR /workspace
EXPOSE 22
CMD ["/opt/openlens/runpod-start.sh"]
