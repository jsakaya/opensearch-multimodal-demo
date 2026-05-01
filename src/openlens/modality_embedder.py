from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .embeddings import FeatureHashEmbedder, audio_evidence_text, mean_pool
from .models import OpenRecord, Patch
from .text import compose_search_text


COMMON_VECTOR_DIM = 384
AUDIO_VECTOR_DIM = 512
QWEN_VECTOR_DIM = 4096
COLPALI_VECTOR_DIM = 128

MODALITY_VECTOR_DIMS = {
    "text_vector": COMMON_VECTOR_DIM,
    "table_vector": COMMON_VECTOR_DIM,
    "audio_vector": AUDIO_VECTOR_DIM,
    "qwen_vector": QWEN_VECTOR_DIM,
    "pdf_vector": COLPALI_VECTOR_DIM,
}

MODALITY_PRIMARY_VECTOR_FIELD = {
    "document": "text_vector",
    "mixed": "text_vector",
    "table": "table_vector",
    "audio": "audio_vector",
    "image": "qwen_vector",
    "video": "qwen_vector",
    "pdf": "pdf_vector",
}

VECTOR_MODALITIES = {
    "text_vector": {"document", "mixed"},
    "table_vector": {"table"},
    "audio_vector": {"audio"},
    "qwen_vector": {"image", "video"},
    "pdf_vector": {"pdf"},
}


@dataclass(frozen=True)
class ModalityEmbeddingBundle:
    vector: list[float]
    text_vector: list[float] = field(default_factory=list)
    table_vector: list[float] = field(default_factory=list)
    audio_vector: list[float] = field(default_factory=list)
    qwen_vector: list[float] = field(default_factory=list)
    pdf_vector: list[float] = field(default_factory=list)
    patch_vectors: list[list[float]] = field(default_factory=list)
    colbert_vectors: list[list[float]] = field(default_factory=list)
    embedding_models: dict[str, str] = field(default_factory=dict)
    primary_vector_field: str = "vector"
    vector_fields: dict[str, int] = field(default_factory=dict)
    chunk_strategy: str = "generic"
    encoder_plan: list[dict[str, Any]] = field(default_factory=list)


class ModalityRouterEmbedder(FeatureHashEmbedder):
    """Route each modality through its strongest evidence representation.

    The class is intentionally usable on CPU-only machines: by default it emits
    deterministic local vectors with the same dimensional contract as the real
    production encoders. Set OPENLENS_USE_REAL_MODALITY_ENCODERS=1 on a GPU
    worker to activate installed Qwen, ColPali, and CLAP providers.
    """

    def __init__(
        self,
        dimension: int = COMMON_VECTOR_DIM,
        qwen_model: str = "qwen8b",
        qwen_batch_size: int = 1,
        qwen_max_frames: int = 32,
        qwen_fps: float = 1.0,
        colpali_model: str = "colpali-v1.3",
        colpali_batch_size: int = 2,
        colpali_max_pages: int = 1,
        colpali_max_patch_vectors: int = 1024,
        colpali_image_timeout_s: float = 20,
    ):
        super().__init__(dimension=dimension)
        self.backend = "modality-router"
        self.model_name = "qwen-video+clap-audio+colpali-docs+bge-text"
        self.text_embedder = FeatureHashEmbedder(COMMON_VECTOR_DIM)
        self.table_embedder = FeatureHashEmbedder(COMMON_VECTOR_DIM)
        self.audio_embedder = FeatureHashEmbedder(AUDIO_VECTOR_DIM)
        self.qwen_fallback = FeatureHashEmbedder(QWEN_VECTOR_DIM)
        self.colpali_fallback = FeatureHashEmbedder(COLPALI_VECTOR_DIM)
        self.qwen_model = qwen_model
        self.clap_model = os.getenv("OPENLENS_CLAP_MODEL", "laion/clap-htsat-unfused")
        self.qwen_batch_size = qwen_batch_size
        self.qwen_max_frames = qwen_max_frames
        self.qwen_fps = qwen_fps
        self.colpali_model = colpali_model
        self.colpali_batch_size = colpali_batch_size
        self.colpali_max_pages = colpali_max_pages
        self.colpali_max_patch_vectors = colpali_max_patch_vectors
        self.colpali_image_timeout_s = colpali_image_timeout_s
        self.use_real = os.getenv("OPENLENS_USE_REAL_MODALITY_ENCODERS", "0").lower() in {"1", "true", "yes", "on"}
        self.use_real_audio = os.getenv("OPENLENS_USE_REAL_AUDIO_ENCODER", "").lower() in {"1", "true", "yes", "on"}
        self.use_real_audio = self.use_real_audio or self.use_real
        self._qwen = None
        self._colpali = None
        self._clap = None

    def prepare_indexed_record(self, record: OpenRecord) -> tuple[list[Patch], ModalityEmbeddingBundle]:
        patches = self.patch_record(record)
        text_vector = self._text_vector(record)
        table_vector = self._table_vector(record) if record.modality == "table" else []
        audio_vector = self._audio_vector(record, patches) if record.modality == "audio" else []
        qwen_vector = self._qwen_vector(record, patches) if record.modality in {"image", "video"} else []
        pdf_vectors = self._colpali_vectors(record, patches) if record.modality == "pdf" else []
        image_colbert = self._colpali_vectors(record, patches) if record.modality == "image" else []
        pdf_vector = mean_pool(pdf_vectors, COLPALI_VECTOR_DIM) if pdf_vectors else []
        primary_field = MODALITY_PRIMARY_VECTOR_FIELD.get(record.modality, "text_vector")
        primary_vector = {
            "text_vector": text_vector,
            "table_vector": table_vector,
            "audio_vector": audio_vector,
            "qwen_vector": qwen_vector,
            "pdf_vector": pdf_vector,
        }.get(primary_field) or text_vector
        common_vector = self.text_embedder.embed_text(
            " ".join(
                part
                for part in [
                    f"modality:{record.modality}",
                    f"primary_vector:{primary_field}",
                    record.title,
                    record.summary,
                    " ".join(record.tags),
                    compose_search_text(record),
                ]
                if part
            )
        )
        patch_vectors = self.text_embedder.embed_patches(patches)
        colbert_vectors = pdf_vectors or image_colbert
        vector_fields = {
            "vector": len(common_vector),
            "text_vector": len(text_vector),
        }
        for name, value in [
            ("table_vector", table_vector),
            ("audio_vector", audio_vector),
            ("qwen_vector", qwen_vector),
            ("pdf_vector", pdf_vector),
        ]:
            if value:
                vector_fields[name] = len(value)
        return patches, ModalityEmbeddingBundle(
            vector=common_vector,
            text_vector=text_vector,
            table_vector=table_vector,
            audio_vector=audio_vector,
            qwen_vector=qwen_vector,
            pdf_vector=pdf_vector,
            patch_vectors=patch_vectors,
            colbert_vectors=colbert_vectors,
            embedding_models=self._embedding_models(record),
            primary_vector_field=primary_field if primary_vector else "text_vector",
            vector_fields=vector_fields,
            chunk_strategy=chunk_strategy(record),
            encoder_plan=encoder_plan(record),
        )

    def embed_text(self, text: str) -> list[float]:
        return self.text_embedder.embed_text(text)

    def embed_query_for_field(self, query: str, vector_field: str) -> list[float]:
        if vector_field == "audio_vector":
            if self.use_real_audio:
                try:
                    return self._audio_embedder().embed_text(query)
                except Exception:
                    pass
            return self.audio_embedder.embed_text(f"audio event or transcript query: {query}")
        if vector_field == "qwen_vector":
            if self.use_real:
                return self._qwen_embedder().embed_text(query)
            return self.qwen_fallback.embed_text(f"qwen video image query: {query}")
        if vector_field == "pdf_vector":
            return mean_pool(self.embed_query_patches(query), COLPALI_VECTOR_DIM)
        if vector_field == "table_vector":
            return self.table_embedder.embed_text(f"sql table row query: {query}")
        return self.text_embedder.embed_text(query)

    def embed_query_patches(self, query: str, max_patches: int = 8) -> list[list[float]]:
        if self.use_real:
            try:
                return self._colpali_embedder().embed_query_patches(query, max_patches=max_patches)
            except Exception:
                pass
        chunks = self.colpali_fallback.embed_query_patches(f"visual document query: {query}", max_patches=max_patches)
        return chunks

    def _text_vector(self, record: OpenRecord) -> list[float]:
        return self.text_embedder.embed_record(record)

    def _table_vector(self, record: OpenRecord) -> list[float]:
        return self.table_embedder.embed_text(f"sql row table {' '.join(record.tags)} {record.title} {record.body}")

    def _audio_vector(self, record: OpenRecord, patches: list[Patch]) -> list[float]:
        text = " ".join([audio_evidence_text(record), *[patch.text for patch in patches]])
        if self.use_real_audio:
            try:
                return self._audio_embedder().embed_record_audio(record, patches, text)
            except Exception:
                pass
        return self.audio_embedder.embed_text(f"clap audio event speech music ambience {text}")

    def _qwen_vector(self, record: OpenRecord, patches: list[Patch]) -> list[float]:
        if self.use_real:
            try:
                return mean_pool(self._qwen_embedder().embed_patches(patches), QWEN_VECTOR_DIM)
            except Exception:
                pass
        return self.qwen_fallback.embed_text(f"qwen multimodal {record.modality} {compose_search_text(record)}")

    def _colpali_vectors(self, record: OpenRecord, patches: list[Patch]) -> list[list[float]]:
        del record
        if self.use_real:
            try:
                return self._colpali_embedder().embed_patches(patches)
            except Exception:
                pass
        return self.colpali_fallback.embed_patches(patches)

    def _qwen_embedder(self):
        if self._qwen is None:
            from .qwen_embedder import QwenMultimodalEmbedder

            self._qwen = QwenMultimodalEmbedder(
                model_name=self.qwen_model,
                dimension=QWEN_VECTOR_DIM,
                batch_size=self.qwen_batch_size,
                max_frames=self.qwen_max_frames,
                fps=self.qwen_fps,
            )
        return self._qwen

    def _colpali_embedder(self):
        if self._colpali is None:
            from .colpali_embedder import ColPaliEmbedder

            self._colpali = ColPaliEmbedder(
                model_name=self.colpali_model,
                dimension=COLPALI_VECTOR_DIM,
                batch_size=self.colpali_batch_size,
                max_pages=self.colpali_max_pages,
                max_patch_vectors=self.colpali_max_patch_vectors,
                image_timeout_s=self.colpali_image_timeout_s,
            )
        return self._colpali

    def _audio_embedder(self):
        if self._clap is None:
            from .audio_embedder import ClapAudioEmbedder

            self._clap = ClapAudioEmbedder(model_name=self.clap_model, dimension=AUDIO_VECTOR_DIM)
        return self._clap

    def _embedding_models(self, record: OpenRecord) -> dict[str, str]:
        models = {
            "text": "bge-m3-compatible-local-feature-hash",
            "table": "row-cell-feature-hash-sql",
            "audio": self.clap_model if self.use_real_audio else "laion-clap-compatible-local-feature-hash",
            "video": self.qwen_model,
            "image": self.qwen_model,
            "pdf": self.colpali_model,
            "late_interaction": self.colpali_model,
        }
        return {"primary": models.get(record.modality, models["text"]), **models}


def chunk_strategy(record: OpenRecord) -> str:
    if record.modality == "video":
        return "sentrysearch-video-30s-overlap-5s-qwen3-vl"
    if record.modality == "audio":
        return "audio-30s-overlap-5s-whisper-transcript-clap-native"
    if record.modality == "pdf":
        return "page-image-colpali-late-interaction"
    if record.modality == "image":
        return "image-qwen3-vl-plus-colpali-patches"
    if record.modality == "table":
        return "sql-row-cell-evidence"
    return "text-semantic-chunks"


def encoder_plan(record: OpenRecord) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = [
        {
            "name": "common_text",
            "field": "vector",
            "model": "BM25 + text dense fallback",
            "purpose": "cross-corpus recall and hybrid fusion",
        }
    ]
    if record.modality == "video":
        plan.append(
            {
                "name": "video_native",
                "field": "qwen_vector",
                "model": "Qwen3-VL-Embedding video",
                "purpose": "SentrySearch-style clip retrieval over sampled video windows",
            }
        )
    elif record.modality == "audio":
        plan.append(
            {
                "name": "audio_native",
                "field": "audio_vector",
                "model": "LAION-CLAP audio/text space plus transcript text",
                "purpose": "speech, music, and environmental sound retrieval",
            }
        )
    elif record.modality == "pdf":
        plan.append(
            {
                "name": "visual_document",
                "field": "pdf_vector",
                "model": "ColPali page image embedding",
                "purpose": "PDF/page recall before late-interaction reranking",
            }
        )
        plan.append(
            {
                "name": "late_interaction",
                "field": "colbert_vectors",
                "model": "ColPali patch vectors",
                "purpose": "token-to-patch reranking with lateInteractionScore",
            }
        )
    elif record.modality == "image":
        plan.append(
            {
                "name": "image_native",
                "field": "qwen_vector",
                "model": "Qwen3-VL-Embedding image",
                "purpose": "text-to-image and image-like retrieval",
            }
        )
        plan.append(
            {
                "name": "visual_patch",
                "field": "colbert_vectors",
                "model": "ColPali-compatible image patches",
                "purpose": "fine visual evidence reranking",
            }
        )
    elif record.modality == "table":
        plan.append(
            {
                "name": "sql_rows",
                "field": "table_vector",
                "model": "row/cell dense embedding plus OpenSearch SQL",
                "purpose": "structured row retrieval and exact SQL filtering",
            }
        )
    else:
        plan.append(
            {
                "name": "text_dense",
                "field": "text_vector",
                "model": "BGE-M3-compatible text dense fallback",
                "purpose": "document and chunk semantic retrieval",
            }
        )
    return plan


def vector_field_for_modality(modality: str | None) -> str:
    if not modality:
        return "vector"
    return MODALITY_PRIMARY_VECTOR_FIELD.get(modality, "vector")


def modalities_for_vector_field(vector_field: str) -> set[str]:
    return VECTOR_MODALITIES.get(vector_field, set())
