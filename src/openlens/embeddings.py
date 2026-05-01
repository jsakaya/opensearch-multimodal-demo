from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np

from .models import OpenRecord
from .text import compose_search_text


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_./:-]*")
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if len(token) > 1 and token not in STOPWORDS]


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


class FeatureHashEmbedder:
    """Deterministic local embedder so the demo runs without hosted models."""

    def __init__(self, dimension: int = 384):
        self.dimension = int(dimension)

    def _feature_vector(self, feature: str) -> np.ndarray:
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "little", signed=False)
        rng = np.random.default_rng(seed)
        return normalize(rng.normal(0, 1, self.dimension).astype(np.float32))

    def embed_text(self, text: str) -> list[float]:
        tokens = tokenize(text)
        if not tokens:
            return [0.0] * self.dimension
        counts = Counter(tokens)
        vector = np.zeros(self.dimension, dtype=np.float32)
        for token, count in counts.items():
            weight = 1.0 + math.log1p(count)
            vector += self._feature_vector(token) * weight
        return normalize(vector).astype(float).tolist()

    def embed_record(self, record: OpenRecord) -> list[float]:
        prefix = f"modality:{record.modality} source:{record.source} title:{record.title} tags:{' '.join(record.tags)}"
        return self.embed_text(prefix + " " + compose_search_text(record))
