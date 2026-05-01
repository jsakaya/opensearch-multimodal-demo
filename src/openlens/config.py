from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]


def _path_from_env(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default))
    return value if value.is_absolute() else ROOT / value


def _vector_dim_from_env() -> int:
    if os.getenv("OPENLENS_VECTOR_DIM"):
        return int(os.environ["OPENLENS_VECTOR_DIM"])
    backend = os.getenv("OPENLENS_EMBEDDING_BACKEND")
    if backend == "qwen":
        return 4096
    if backend == "colpali":
        return 128
    return 384


def load_project_env() -> None:
    load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    opensearch_url: str = field(default_factory=lambda: os.getenv("OPENSEARCH_URL", "http://localhost:9200"))
    opensearch_index: str = field(default_factory=lambda: os.getenv("OPENSEARCH_INDEX", "openlens_multimodal"))
    opensearch_timeout_s: float = field(default_factory=lambda: float(os.getenv("OPENSEARCH_TIMEOUT_S", "30")))
    vector_dim: int = field(default_factory=_vector_dim_from_env)
    embedding_backend: str = field(default_factory=lambda: os.getenv("OPENLENS_EMBEDDING_BACKEND", "feature-hash"))
    qwen_model: str = field(default_factory=lambda: os.getenv("OPENLENS_QWEN_MODEL", "qwen8b"))
    qwen_batch_size: int = field(default_factory=lambda: int(os.getenv("OPENLENS_QWEN_BATCH_SIZE", "1")))
    qwen_max_frames: int = field(default_factory=lambda: int(os.getenv("OPENLENS_QWEN_MAX_FRAMES", "32")))
    qwen_fps: float = field(default_factory=lambda: float(os.getenv("OPENLENS_QWEN_FPS", "1.0")))
    colpali_model: str = field(default_factory=lambda: os.getenv("OPENLENS_COLPALI_MODEL", "colpali-v1.3"))
    colpali_batch_size: int = field(default_factory=lambda: int(os.getenv("OPENLENS_COLPALI_BATCH_SIZE", "2")))
    colpali_max_pages: int = field(default_factory=lambda: int(os.getenv("OPENLENS_COLPALI_MAX_PAGES", "1")))
    colpali_max_patch_vectors: int = field(
        default_factory=lambda: int(os.getenv("OPENLENS_COLPALI_MAX_PATCH_VECTORS", "1024"))
    )
    colpali_image_timeout_s: float = field(
        default_factory=lambda: float(os.getenv("OPENLENS_COLPALI_IMAGE_TIMEOUT_S", "20"))
    )
    require_opensearch: bool = field(default_factory=lambda: os.getenv("OPENLENS_REQUIRE_OPENSEARCH", "1") != "0")
    docs_path: Path = field(default_factory=lambda: _path_from_env("OPENLENS_DOCS", "data/processed/open_corpus.jsonl"))
    embedded_docs_path: Path = field(
        default_factory=lambda: _path_from_env("OPENLENS_EMBEDDED_DOCS", "data/processed/open_corpus_embedded.jsonl")
    )
    user_agent: str = field(default_factory=lambda: os.getenv("OPENLENS_USER_AGENT", "openlens-opensearch-demo/0.1"))


def get_settings() -> Settings:
    load_project_env()
    return Settings()
