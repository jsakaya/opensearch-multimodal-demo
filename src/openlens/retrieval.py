from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from opensearchpy import OpenSearch
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import Settings
from .data import read_jsonl
from .embeddings import FeatureHashEmbedder
from .indexer import check_status, make_client, prepare_record
from .models import OpenRecord
from .text import excerpt_for


SearchMode = Literal["hybrid", "keyword", "vector"]


@dataclass
class SearchHit:
    doc_id: str
    rank: int
    score: float
    method: str
    doc: dict[str, Any]
    excerpt: str
    components: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "rank": self.rank,
            "score": self.score,
            "method": self.method,
            "excerpt": self.excerpt,
            "components": self.components,
            "source": self.doc.get("source"),
            "source_id": self.doc.get("source_id"),
            "source_url": self.doc.get("source_url"),
            "modality": self.doc.get("modality"),
            "title": self.doc.get("title"),
            "summary": self.doc.get("summary"),
            "license": self.doc.get("license"),
            "license_url": self.doc.get("license_url"),
            "attribution": self.doc.get("attribution"),
            "tags": self.doc.get("tags") or [],
            "facets": self.doc.get("facets") or {},
            "table": self.doc.get("table") or {},
            "assets": self.doc.get("assets") or [],
        }


@dataclass
class SearchResponse:
    query: str
    mode: SearchMode
    retriever: str
    latency_ms: float
    hits: list[SearchHit]
    facets: dict[str, dict[str, int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "mode": self.mode,
            "retriever": self.retriever,
            "latency_ms": round(self.latency_ms, 2),
            "total": len(self.hits),
            "hits": [hit.to_dict() for hit in self.hits],
            "facets": self.facets,
        }


def rrf_fuse(result_lists: list[list[SearchHit]], top_k: int, k: int = 60) -> list[SearchHit]:
    by_id: dict[str, SearchHit] = {}
    scores: dict[str, float] = {}
    components: dict[str, dict[str, float]] = {}
    for results in result_lists:
        for hit in results:
            scores[hit.doc_id] = scores.get(hit.doc_id, 0.0) + 1.0 / (k + hit.rank)
            components.setdefault(hit.doc_id, {})[hit.method] = hit.score
            by_id.setdefault(hit.doc_id, hit)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    fused: list[SearchHit] = []
    for rank, (doc_id, score) in enumerate(ordered, start=1):
        hit = by_id[doc_id]
        fused.append(
            SearchHit(
                doc_id=doc_id,
                rank=rank,
                score=float(score),
                method="hybrid",
                doc=hit.doc,
                excerpt=hit.excerpt,
                components=components.get(doc_id, {}),
            )
        )
    return fused


def facet_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    facets = {"modality": {}, "source": {}}
    for row in rows:
        for name in facets:
            value = str(row.get(name) or "unknown")
            facets[name][value] = facets[name].get(value, 0) + 1
    return facets


class LocalRetriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = FeatureHashEmbedder(settings.vector_dim)
        path = settings.embedded_docs_path if settings.embedded_docs_path.exists() else settings.docs_path
        rows = read_jsonl(path)
        self.docs: list[dict[str, Any]] = []
        for row in rows:
            if "vector" in row and "search_text" in row:
                self.docs.append(row)
            else:
                record = prepare_record(OpenRecord.model_validate(row), self.embedder)
                self.docs.append(record.model_dump(mode="json"))
        self.texts = [doc.get("search_text") or "" for doc in self.docs]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000)
        self.tfidf = self.vectorizer.fit_transform(self.texts) if self.texts else None

    def search(
        self,
        query: str,
        mode: SearchMode = "hybrid",
        top_k: int = 12,
        candidate_k: int = 80,
        modality: str | None = None,
        source: str | None = None,
    ) -> SearchResponse:
        start = time.perf_counter()
        docs = self._filtered_docs(modality=modality, source=source)
        if mode == "keyword":
            hits = self._keyword(query, docs, top_k)
        elif mode == "vector":
            hits = self._vector(query, docs, top_k)
        else:
            hits = rrf_fuse(
                [
                    self._keyword(query, docs, min(candidate_k, len(docs))),
                    self._vector(query, docs, min(candidate_k, len(docs))),
                ],
                top_k=top_k,
            )
        return SearchResponse(
            query=query,
            mode=mode,
            retriever="local",
            latency_ms=(time.perf_counter() - start) * 1000,
            hits=hits,
            facets=facet_counts(docs),
        )

    def _filtered_docs(self, modality: str | None, source: str | None) -> list[dict[str, Any]]:
        docs = self.docs
        if modality:
            docs = [doc for doc in docs if doc.get("modality") == modality]
        if source:
            docs = [doc for doc in docs if doc.get("source") == source]
        return docs

    def _keyword(self, query: str, docs: list[dict[str, Any]], top_k: int) -> list[SearchHit]:
        if not docs or self.tfidf is None:
            return []
        doc_indexes = [self.docs.index(doc) for doc in docs]
        q = self.vectorizer.transform([query])
        scores = (self.tfidf[doc_indexes] @ q.T).toarray().ravel()
        order = np.argsort(scores)[::-1][:top_k]
        hits = []
        for rank, local_index in enumerate(order, start=1):
            score = float(scores[local_index])
            if score <= 0:
                continue
            doc = docs[int(local_index)]
            hits.append(self._result(doc, rank, score, "keyword", query))
        return hits

    def _vector(self, query: str, docs: list[dict[str, Any]], top_k: int) -> list[SearchHit]:
        if not docs:
            return []
        q = np.asarray(self.embedder.embed_text(query), dtype=np.float32)
        scores = [float(np.dot(q, np.asarray(doc.get("vector", []), dtype=np.float32))) for doc in docs]
        order = np.argsort(np.asarray(scores))[::-1][:top_k]
        return [self._result(docs[int(i)], rank, float(scores[int(i)]), "vector", query) for rank, i in enumerate(order, start=1)]

    def _result(self, doc: dict[str, Any], rank: int, score: float, method: str, query: str) -> SearchHit:
        return SearchHit(
            doc_id=str(doc.get("doc_id")),
            rank=rank,
            score=score,
            method=method,
            doc=doc,
            excerpt=excerpt_for(query, doc.get("search_text") or doc.get("body") or doc.get("summary") or ""),
            components={method: score},
        )


class OpenSearchRetriever:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client: OpenSearch = make_client(settings)
        self.embedder = FeatureHashEmbedder(settings.vector_dim)

    def search(
        self,
        query: str,
        mode: SearchMode = "hybrid",
        top_k: int = 12,
        candidate_k: int = 80,
        modality: str | None = None,
        source: str | None = None,
    ) -> SearchResponse:
        start = time.perf_counter()
        filters = self._filters(modality=modality, source=source)
        if mode == "keyword":
            hits = self._keyword(query, top_k, filters)
        elif mode == "vector":
            hits = self._vector(query, top_k, filters)
        else:
            hits = rrf_fuse([self._keyword(query, candidate_k, filters), self._vector(query, candidate_k, filters)], top_k)
        facets = self._facet_counts(query, filters)
        return SearchResponse(
            query=query,
            mode=mode,
            retriever="opensearch",
            latency_ms=(time.perf_counter() - start) * 1000,
            hits=hits,
            facets=facets,
        )

    def _filters(self, modality: str | None, source: str | None) -> list[dict[str, Any]]:
        filters = []
        if modality:
            filters.append({"term": {"modality": modality}})
        if source:
            filters.append({"term": {"source": source}})
        return filters

    def _keyword(self, query: str, top_k: int, filters: list[dict[str, Any]]) -> list[SearchHit]:
        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^5", "summary^3", "body^2", "search_text"],
                                "type": "best_fields",
                            }
                        }
                    ],
                    "filter": filters,
                }
            },
            "highlight": {"fields": {"title": {}, "summary": {}, "body": {}, "search_text": {}}},
            "_source": {"excludes": ["vector"]},
        }
        response = self.client.search(index=self.settings.opensearch_index, body=body)
        return [self._hit_to_result(hit, rank, "keyword", query) for rank, hit in enumerate(response["hits"]["hits"], start=1)]

    def _vector(self, query: str, top_k: int, filters: list[dict[str, Any]]) -> list[SearchHit]:
        vector = self.embedder.embed_text(query)
        knn: dict[str, Any] = {"vector": vector, "k": top_k}
        if filters:
            knn["filter"] = {"bool": {"filter": filters}}
        body = {"size": top_k, "query": {"knn": {"vector": knn}}, "_source": {"excludes": ["vector"]}}
        response = self.client.search(index=self.settings.opensearch_index, body=body)
        return [self._hit_to_result(hit, rank, "vector", query) for rank, hit in enumerate(response["hits"]["hits"], start=1)]

    def _facet_counts(self, query: str, filters: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
        body = {
            "size": 0,
            "query": {
                "bool": {
                    "must": [
                        {"multi_match": {"query": query, "fields": ["title^5", "summary^3", "body^2", "search_text"]}}
                    ],
                    "filter": filters,
                }
            },
            "aggs": {
                "modality": {"terms": {"field": "modality", "size": 12}},
                "source": {"terms": {"field": "source", "size": 12}},
            },
        }
        response = self.client.search(index=self.settings.opensearch_index, body=body)
        return {
            name: {
                bucket["key"]: int(bucket["doc_count"])
                for bucket in response.get("aggregations", {}).get(name, {}).get("buckets", [])
            }
            for name in ("modality", "source")
        }

    def _hit_to_result(self, hit: dict[str, Any], rank: int, method: str, query: str) -> SearchHit:
        source = hit.get("_source", {})
        excerpt = ""
        for values in (hit.get("highlight") or {}).values():
            if values:
                excerpt = values[0].replace("<em>", "").replace("</em>", "")
                break
        if not excerpt:
            excerpt = excerpt_for(query, source.get("search_text") or source.get("body") or source.get("summary") or "")
        score = float(hit.get("_score") or 0.0)
        return SearchHit(str(source.get("doc_id") or hit.get("_id")), rank, score, method, source, excerpt, {method: score})


def make_retriever(settings: Settings, prefer_opensearch: bool = True) -> LocalRetriever | OpenSearchRetriever:
    status = check_status(settings)
    if prefer_opensearch and status.available and status.doc_count > 0:
        return OpenSearchRetriever(settings)
    return LocalRetriever(settings)
