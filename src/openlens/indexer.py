from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import urlparse

from opensearchpy import OpenSearch, helpers

from .config import Settings
from .data import utc_now, write_jsonl
from .embeddings import FeatureHashEmbedder, mean_pool
from .modality_embedder import MODALITY_VECTOR_DIMS
from .models import IndexedRecord, OpenRecord
from .qwen_embedder import make_embedder
from .text import compose_search_text

VECTOR_SOURCE_FIELDS = [
    "vector",
    "text_vector",
    "table_vector",
    "audio_vector",
    "qwen_vector",
    "pdf_vector",
]


@dataclass(frozen=True)
class OpenSearchStatus:
    available: bool
    detail: str
    doc_count: int = 0


def make_client(settings: Settings) -> OpenSearch:
    parsed = urlparse(settings.opensearch_url)
    return OpenSearch(
        hosts=[settings.opensearch_url],
        use_ssl=parsed.scheme == "https",
        verify_certs=False,
        ssl_show_warn=False,
        timeout=settings.opensearch_timeout_s,
        max_retries=2,
        retry_on_timeout=True,
    )


def check_status(settings: Settings) -> OpenSearchStatus:
    try:
        client = make_client(settings)
        info = client.info()
        version = info.get("version", {}).get("number", "unknown")
        doc_count = 0
        if client.indices.exists(index=settings.opensearch_index):
            doc_count = int(client.count(index=settings.opensearch_index).get("count", 0))
        return OpenSearchStatus(True, f"OpenSearch {version} at {settings.opensearch_url}", doc_count=doc_count)
    except Exception as exc:
        return OpenSearchStatus(False, f"{type(exc).__name__}: {exc}")


def index_mapping(dimension: int) -> dict[str, Any]:
    def knn_vector(vector_dimension: int) -> dict[str, Any]:
        return {
            "type": "knn_vector",
            "dimension": vector_dimension,
            "method": {
                "name": "hnsw",
                "engine": "lucene",
                "space_type": "innerproduct",
                "parameters": {"ef_construction": 128, "m": 24},
            },
        }

    return {
        "settings": {
            "index.knn": True,
            "index.knn.algo_param.ef_search": 128,
            "index.refresh_interval": "1s",
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "openlens_text": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding", "porter_stem"],
                    }
                }
            },
        },
        "mappings": {
            "dynamic": "false",
            "properties": {
                "doc_id": {"type": "keyword"},
                "source": {"type": "keyword"},
                "source_id": {"type": "keyword"},
                "source_url": {"type": "keyword", "index": False},
                "modality": {"type": "keyword"},
                "title": {"type": "text", "analyzer": "openlens_text", "fields": {"raw": {"type": "keyword"}}},
                "summary": {"type": "text", "analyzer": "openlens_text"},
                "body": {"type": "text", "analyzer": "openlens_text"},
                "search_text": {"type": "text", "analyzer": "openlens_text"},
                "license": {"type": "keyword"},
                "license_url": {"type": "keyword", "index": False},
                "attribution": {"type": "text"},
                "language": {"type": "keyword"},
                "published_at": {"type": "date", "ignore_malformed": True},
                "updated_at": {"type": "date", "ignore_malformed": True},
                "indexed_at": {"type": "date"},
                "patch_count": {"type": "integer"},
                "embedding_backend": {"type": "keyword"},
                "embedding_model": {"type": "keyword"},
                "primary_vector_field": {"type": "keyword"},
                "chunk_strategy": {"type": "keyword"},
                "tags": {"type": "keyword"},
                "facets": {"type": "object", "enabled": False},
                "table": {"type": "object", "enabled": False},
                "assets": {"type": "object", "enabled": False},
                "patches": {"type": "object", "enabled": False},
                "patch_vectors": {"type": "object", "enabled": False},
                "colbert_vectors": {"type": "object", "enabled": False},
                "embedding_models": {"type": "object", "enabled": False},
                "vector_fields": {"type": "object", "enabled": False},
                "encoder_plan": {"type": "object", "enabled": False},
                "patch_vector_count": {"type": "integer"},
                "vector": knn_vector(dimension),
                **{field: knn_vector(field_dimension) for field, field_dimension in MODALITY_VECTOR_DIMS.items()},
            }
        },
    }


def prepare_record(record: OpenRecord, embedder: FeatureHashEmbedder) -> IndexedRecord:
    if hasattr(embedder, "prepare_indexed_record"):
        patches, bundle = embedder.prepare_indexed_record(record)  # type: ignore[attr-defined]
        return IndexedRecord(
            **record.model_dump(),
            search_text=compose_search_text(record),
            vector=bundle.vector,
            text_vector=bundle.text_vector,
            table_vector=bundle.table_vector,
            audio_vector=bundle.audio_vector,
            qwen_vector=bundle.qwen_vector,
            pdf_vector=bundle.pdf_vector,
            patches=patches,
            patch_vectors=bundle.patch_vectors,
            colbert_vectors=bundle.colbert_vectors,
            patch_count=len(patches),
            patch_vector_count=len(bundle.colbert_vectors) or len(bundle.patch_vectors),
            embedding_backend=getattr(embedder, "backend", "feature-hash"),
            embedding_model=getattr(embedder, "model_name", "feature-hash"),
            embedding_models=bundle.embedding_models,
            primary_vector_field=bundle.primary_vector_field,
            vector_fields=bundle.vector_fields,
            chunk_strategy=bundle.chunk_strategy,
            encoder_plan=bundle.encoder_plan,
            indexed_at=utc_now(),
        )
    patches = embedder.patch_record(record)
    patch_vectors = embedder.embed_patches(patches)
    vector = mean_pool(patch_vectors or [embedder.embed_record(record)], embedder.dimension)
    return IndexedRecord(
        **record.model_dump(),
        search_text=compose_search_text(record),
        vector=vector,
        patches=patches,
        patch_vectors=patch_vectors,
        colbert_vectors=patch_vectors,
        patch_count=len(patches),
        patch_vector_count=len(patch_vectors),
        embedding_backend=getattr(embedder, "backend", "feature-hash"),
        embedding_model=getattr(embedder, "model_name", "feature-hash"),
        primary_vector_field="vector",
        vector_fields={"vector": len(vector)},
        chunk_strategy="feature-hash-patches",
        encoder_plan=[
            {
                "name": "feature_hash",
                "field": "vector",
                "model": getattr(embedder, "model_name", "feature-hash"),
                "purpose": "portable local recall and smoke testing",
            }
        ],
        indexed_at=utc_now(),
    )


def prepare_records(records: Iterable[OpenRecord], embedder: FeatureHashEmbedder) -> list[IndexedRecord]:
    return [prepare_record(record, embedder) for record in records]


def recreate_index(client: OpenSearch, index_name: str, dimension: int) -> None:
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
    client.indices.create(index=index_name, body=index_mapping(dimension))


def bulk_index(client: OpenSearch, index_name: str, records: Iterable[IndexedRecord], refresh: bool | str = False) -> int:
    actions = (
        {
            "_op_type": "index",
            "_index": index_name,
            "_id": record.doc_id,
            "_source": opensearch_source(record),
        }
        for record in records
    )
    successes = 0
    for ok, _result in helpers.streaming_bulk(client, actions, chunk_size=500, refresh=refresh, request_timeout=120):
        if ok:
            successes += 1
    return successes


def opensearch_source(record: IndexedRecord) -> dict[str, Any]:
    source = record.model_dump(mode="json")
    for field in VECTOR_SOURCE_FIELDS:
        value = source.get(field)
        if not isinstance(value, list) or not value:
            source.pop(field, None)
    return source


def embed_and_optionally_index(
    settings: Settings,
    records: list[OpenRecord],
    recreate: bool = True,
    skip_opensearch: bool = False,
) -> tuple[list[IndexedRecord], OpenSearchStatus]:
    embedder = make_embedder(
        settings.embedding_backend,
        settings.vector_dim,
        settings.qwen_model,
        batch_size=settings.qwen_batch_size,
        max_frames=settings.qwen_max_frames,
        fps=settings.qwen_fps,
        colpali_batch_size=settings.colpali_batch_size,
        colpali_model=settings.colpali_model,
        colpali_max_pages=settings.colpali_max_pages,
        colpali_max_patch_vectors=settings.colpali_max_patch_vectors,
        colpali_image_timeout_s=settings.colpali_image_timeout_s,
    )
    indexed = prepare_records(records, embedder)
    write_jsonl(settings.embedded_docs_path, [record.model_dump(mode="json") for record in indexed])
    if skip_opensearch:
        return indexed, OpenSearchStatus(False, "Skipped OpenSearch indexing by request.")
    status = check_status(settings)
    if not status.available:
        return indexed, status
    client = make_client(settings)
    if recreate:
        recreate_index(client, settings.opensearch_index, settings.vector_dim)
    count = bulk_index(client, settings.opensearch_index, indexed, refresh=True)
    return indexed, OpenSearchStatus(True, f"Indexed {count:,} documents into {settings.opensearch_index}", doc_count=count)
