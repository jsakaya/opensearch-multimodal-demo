from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import urlparse

from opensearchpy import OpenSearch, helpers

from .config import Settings
from .data import utc_now, write_jsonl
from .embeddings import FeatureHashEmbedder
from .models import IndexedRecord, OpenRecord
from .text import compose_search_text


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
                "tags": {"type": "keyword"},
                "facets": {"type": "object", "enabled": False},
                "table": {"type": "object", "enabled": False},
                "assets": {"type": "object", "enabled": False},
                "vector": {
                    "type": "knn_vector",
                    "dimension": dimension,
                    "method": {
                        "name": "hnsw",
                        "engine": "lucene",
                        "space_type": "innerproduct",
                        "parameters": {"ef_construction": 128, "m": 24},
                    },
                },
            }
        },
    }


def prepare_record(record: OpenRecord, embedder: FeatureHashEmbedder) -> IndexedRecord:
    return IndexedRecord(
        **record.model_dump(),
        search_text=compose_search_text(record),
        vector=embedder.embed_record(record),
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
            "_source": record.model_dump(mode="json"),
        }
        for record in records
    )
    successes = 0
    for ok, _result in helpers.streaming_bulk(client, actions, chunk_size=500, refresh=refresh, request_timeout=120):
        if ok:
            successes += 1
    return successes


def embed_and_optionally_index(
    settings: Settings,
    records: list[OpenRecord],
    recreate: bool = True,
    skip_opensearch: bool = False,
) -> tuple[list[IndexedRecord], OpenSearchStatus]:
    embedder = FeatureHashEmbedder(settings.vector_dim)
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
