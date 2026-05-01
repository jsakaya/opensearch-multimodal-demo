from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np

from .data import stable_id
from .models import OpenRecord, Patch
from .text import compose_search_text
from .video import expected_chunk_spans


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
        self.backend = "feature-hash"
        self.model_name = "feature-hash"

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

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    def patch_record(self, record: OpenRecord, max_patches: int = 12, patch_chars: int = 560) -> list[Patch]:
        patches: list[Patch] = []

        def add(
            kind: str,
            text: str,
            page: int | None = None,
            start_s: float | None = None,
            end_s: float | None = None,
            asset_url: str = "",
        ) -> None:
            text = " ".join(str(text or "").split())
            if not text:
                return
            ordinal = len(patches)
            patches.append(
                Patch(
                    patch_id=f"{record.doc_id}:p{ordinal:03d}-{stable_id(kind, text)}",
                    kind=kind,
                    ordinal=ordinal,
                    text=text[:patch_chars],
                    page=page,
                    start_s=start_s,
                    end_s=end_s,
                    asset_url=asset_url,
                )
            )

        add("record_header", f"{record.title}. {record.summary}")
        if record.modality in {"pdf", "document"}:
            for idx, chunk in enumerate(chunk_text(record.body or record.summary, patch_chars)):
                add("pdf_page_patch" if record.modality == "pdf" else "text_patch", chunk, page=idx + 1)
        elif record.modality == "image":
            add("visual_caption", f"{record.title}. {record.summary}. {' '.join(record.tags)}")
            for asset in record.assets[:3]:
                add(
                    "visual_asset",
                    f"{asset.kind} {asset.mime_type} width {asset.width} height {asset.height} {asset.url}",
                    asset_url=asset.url,
                )
        elif record.modality == "video":
            chunks = chunk_text(record.body or record.summary or record.title, patch_chars) or [record.title]
            duration_s = next((asset.duration_s for asset in record.assets if asset.duration_s), None)
            spans = expected_chunk_spans(duration_s or len(chunks) * 30, chunk_duration_s=30, overlap_s=5)
            for idx, chunk in enumerate(chunks):
                span = spans[min(idx, len(spans) - 1)]
                add(
                    "video_semantic_chunk",
                    chunk,
                    page=idx + 1,
                    start_s=span.start_s,
                    end_s=span.end_s,
                    asset_url=record.assets[0].url if record.assets else "",
                )
        elif record.modality == "audio":
            chunks = chunk_text(record.body or record.summary or record.title, patch_chars) or [record.title]
            for idx, chunk in enumerate(chunks):
                add(
                    "audio_semantic_chunk",
                    chunk,
                    page=idx + 1,
                    start_s=float(idx * 30),
                    end_s=float((idx + 1) * 30),
                    asset_url=record.assets[0].url if record.assets else "",
                )
        elif record.modality == "table":
            for idx, (key, value) in enumerate(record.table.items()):
                add("table_cell", f"{key.replace('_', ' ')}: {value}", page=idx + 1)
        else:
            for idx, chunk in enumerate(chunk_text(compose_search_text(record), patch_chars)):
                add("mixed_patch", chunk, page=idx + 1)
        if len(patches) < 2:
            add("record_body", compose_search_text(record))
        return patches[:max_patches]

    def embed_patches(self, patches: list[Patch]) -> list[list[float]]:
        return self.embed_texts([patch.text for patch in patches])

    def embed_query_patches(self, query: str, max_patches: int = 8) -> list[list[float]]:
        tokens = tokenize(query)
        chunks = [query]
        if len(tokens) > 4:
            chunks.extend(" ".join(tokens[start : start + 5]) for start in range(0, len(tokens), 4))
        return self.embed_texts(chunks[:max_patches])


def chunk_text(text: str, patch_chars: int = 560) -> list[str]:
    text = " ".join(str(text or "").split())
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + patch_chars)
        if end < len(text):
            split = text.rfind(" ", start + int(patch_chars * 0.65), end)
            if split > start:
                end = split
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end, start + 1)
    return chunks


def mean_pool(vectors: list[list[float]], dimension: int) -> list[float]:
    if not vectors:
        return [0.0] * dimension
    arr = np.asarray(vectors, dtype=np.float32)
    return normalize(arr.mean(axis=0)).astype(float).tolist()


def late_interaction_score(query_vectors: list[list[float]], doc_vectors: list[list[float]]) -> float:
    if not query_vectors or not doc_vectors:
        return 0.0
    q = np.asarray(query_vectors, dtype=np.float32)
    d = np.asarray(doc_vectors, dtype=np.float32)
    sims = q @ d.T
    return float(sims.max(axis=1).sum())
