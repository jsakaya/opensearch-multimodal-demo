from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]


def _path_from_env(name: str, default: str) -> Path:
    value = Path(os.getenv(name, default))
    return value if value.is_absolute() else ROOT / value


def load_project_env() -> None:
    load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    opensearch_url: str = os.getenv("OPENSEARCH_URL", "http://localhost:9200")
    opensearch_index: str = os.getenv("OPENSEARCH_INDEX", "openlens_multimodal")
    opensearch_timeout_s: float = float(os.getenv("OPENSEARCH_TIMEOUT_S", "30"))
    vector_dim: int = int(os.getenv("OPENLENS_VECTOR_DIM", "384"))
    docs_path: Path = _path_from_env("OPENLENS_DOCS", "data/processed/open_corpus.jsonl")
    embedded_docs_path: Path = _path_from_env("OPENLENS_EMBEDDED_DOCS", "data/processed/open_corpus_embedded.jsonl")
    user_agent: str = os.getenv("OPENLENS_USER_AGENT", "openlens-opensearch-demo/0.1")


def get_settings() -> Settings:
    load_project_env()
    return Settings()
