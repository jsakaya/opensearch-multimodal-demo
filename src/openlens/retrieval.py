from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from opensearchpy import OpenSearch
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import Settings
from .data import read_jsonl
from .embeddings import late_interaction_score, mean_pool
from .indexer import check_status, make_client, prepare_record
from .models import OpenRecord
from .qwen_embedder import make_embedder
from .text import excerpt_for


SearchMode = Literal["hybrid", "keyword", "vector", "lir", "sql"]


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
            "patches": self.doc.get("patches") or [],
            "patch_count": self.doc.get("patch_count") or len(self.doc.get("patches") or []),
            "patch_vector_count": self.doc.get("patch_vector_count") or len(_late_vectors(self.doc, "patch_vectors")),
            "embedding_backend": self.doc.get("embedding_backend"),
            "embedding_model": self.doc.get("embedding_model"),
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
        self.embedder = make_embedder(
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
        path = settings.embedded_docs_path if settings.embedded_docs_path.exists() else settings.docs_path
        rows = read_jsonl(path)
        self.docs: list[dict[str, Any]] = []
        for row in rows:
            if "vector" in row and "search_text" in row:
                self.docs.append(row)
            else:
                record = prepare_record(OpenRecord.model_validate(row), self.embedder)
                self.docs.append(record.model_dump(mode="json"))
        self.doc_positions = {id(doc): index for index, doc in enumerate(self.docs)}
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
        elif mode == "lir":
            hits = self._lir(query, docs, top_k, candidate_k)
        elif mode == "sql":
            hits = self._keyword(query, [doc for doc in docs if doc.get("modality") == "table"], top_k)
            for hit in hits:
                hit.method = "sql"
                hit.components = {"sql": hit.score}
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
        doc_indexes = [self.doc_positions[id(doc)] for doc in docs]
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

    def _lir(self, query: str, docs: list[dict[str, Any]], top_k: int, candidate_k: int) -> list[SearchHit]:
        if not docs:
            return []
        candidates = rrf_fuse(
            [
                self._keyword(query, docs, min(candidate_k, len(docs))),
                self._vector(query, docs, min(candidate_k, len(docs))),
            ],
            top_k=min(candidate_k, len(docs)),
        )
        query_vectors = self.embedder.embed_query_patches(query)
        scored: list[tuple[float, SearchHit]] = []
        for candidate in candidates:
            doc_vectors = _late_vectors(candidate.doc, _late_vector_field(self.settings.embedding_backend))
            score = late_interaction_score(query_vectors, doc_vectors)
            if score <= 0:
                score = candidate.score
            scored.append((score, candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        hits: list[SearchHit] = []
        for rank, (score, candidate) in enumerate(scored[:top_k], start=1):
            doc = candidate.doc
            hits.append(
                SearchHit(
                    doc_id=candidate.doc_id,
                    rank=rank,
                    score=score,
                    method="lir",
                    doc=doc,
                    excerpt=best_patch_excerpt(query, doc),
                    components={**candidate.components, "late_interaction": score},
                )
            )
        return hits

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
        self.embedder = make_embedder(
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
        elif mode == "lir":
            hits = self._lir(query, top_k, candidate_k, filters)
        elif mode == "sql":
            hits = self._sql(query, top_k, filters)
        else:
            hits = rrf_fuse([self._keyword(query, candidate_k, filters), self._vector(query, candidate_k, filters)], top_k)
        return SearchResponse(
            query=query,
            mode=mode,
            retriever="opensearch",
            latency_ms=(time.perf_counter() - start) * 1000,
            hits=hits,
            facets=facet_counts([hit.doc for hit in hits]),
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
            "_source": {"excludes": ["vector", "patch_vectors", "colbert_vectors"]},
        }
        response = self.client.search(index=self.settings.opensearch_index, body=body)
        return [self._hit_to_result(hit, rank, "keyword", query) for rank, hit in enumerate(response["hits"]["hits"], start=1)]

    def _vector(self, query: str, top_k: int, filters: list[dict[str, Any]]) -> list[SearchHit]:
        vector = self.embedder.embed_text(query)
        knn: dict[str, Any] = {"vector": vector, "k": top_k}
        if filters:
            knn["filter"] = {"bool": {"filter": filters}}
        body = {
            "size": top_k,
            "query": {"knn": {"vector": knn}},
            "_source": {"excludes": ["vector", "patch_vectors", "colbert_vectors"]},
        }
        response = self.client.search(index=self.settings.opensearch_index, body=body)
        return [self._hit_to_result(hit, rank, "vector", query) for rank, hit in enumerate(response["hits"]["hits"], start=1)]

    def _sql(self, query: str, top_k: int, filters: list[dict[str, Any]]) -> list[SearchHit]:
        table_filters = [*filters, {"term": {"modality": "table"}}]
        if _is_sql_statement(query):
            try:
                return self._sql_plugin(query, top_k)
            except Exception:
                pass
        body = {
            "size": top_k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^5", "summary^3", "body^2", "search_text", "tags^2"],
                                "type": "best_fields",
                            }
                        }
                    ],
                    "filter": table_filters,
                }
            },
            "highlight": {"fields": {"title": {}, "summary": {}, "body": {}, "search_text": {}}},
            "_source": {"excludes": ["vector", "patch_vectors", "colbert_vectors"]},
        }
        response = self.client.search(index=self.settings.opensearch_index, body=body)
        return [self._hit_to_result(hit, rank, "sql", query) for rank, hit in enumerate(response["hits"]["hits"], start=1)]

    def _sql_plugin(self, query: str, top_k: int) -> list[SearchHit]:
        sql = _rewrite_sql_index(query, self.settings.opensearch_index, top_k)
        response = self.client.transport.perform_request(
            "POST",
            "/_plugins/_sql",
            body={"query": sql},
        )
        columns = [column.get("alias") or column.get("name") for column in response.get("schema", [])]
        hits: list[SearchHit] = []
        for rank, row in enumerate(response.get("datarows", [])[:top_k], start=1):
            table = dict(zip(columns, row, strict=False))
            doc_id = f"sql-row-{rank}"
            title = table.get("pl_name") or table.get("title") or table.get("doc_id") or f"SQL row {rank}"
            doc = {
                "doc_id": doc_id,
                "source": "OpenSearch SQL",
                "source_id": doc_id,
                "source_url": "",
                "modality": "table",
                "title": str(title),
                "summary": "OpenSearch SQL result row",
                "license": "",
                "tags": ["sql", "opensearch"],
                "facets": {},
                "table": table,
                "assets": [],
                "patches": [],
                "patch_count": 0,
                "embedding_backend": self.settings.embedding_backend,
                "embedding_model": self.settings.colpali_model
                if self.settings.embedding_backend == "colpali"
                else self.settings.qwen_model,
            }
            hits.append(SearchHit(doc_id, rank, 1.0 / rank, "sql", doc, str(table)[:360], {"sql": 1.0 / rank}))
        return hits

    def _lir(self, query: str, top_k: int, candidate_k: int, filters: list[dict[str, Any]]) -> list[SearchHit]:
        query_vectors = self.embedder.embed_query_patches(query)
        query_vector = mean_pool(query_vectors, self.embedder.dimension)
        vector_field = _late_vector_field(self.settings.embedding_backend)
        try:
            return self._native_lir(query, query_vector, query_vectors, top_k, candidate_k, filters, vector_field)
        except Exception:
            return self._client_lir(query, query_vectors, top_k, candidate_k, filters, vector_field)

    def _native_lir(
        self,
        query: str,
        query_vector: list[float],
        query_vectors: list[list[float]],
        top_k: int,
        candidate_k: int,
        filters: list[dict[str, Any]],
        vector_field: str,
    ) -> list[SearchHit]:
        knn: dict[str, Any] = {"vector": query_vector, "k": candidate_k}
        if filters:
            knn["filter"] = {"bool": {"filter": filters}}
        body = {
            "size": top_k,
            "query": {"knn": {"vector": knn}},
            "rescore": {
                "window_size": candidate_k,
                "query": {
                    "query_weight": 0.0,
                    "rescore_query_weight": 1.0,
                    "rescore_query": {
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": f"lateInteractionScore(params.query_vector, '{vector_field}', params._source)",
                                "params": {"query_vector": query_vectors},
                            },
                        }
                    },
                },
            },
            "_source": {"excludes": ["vector", "patch_vectors", "colbert_vectors"]},
        }
        response = self.client.search(index=self.settings.opensearch_index, body=body, request_timeout=120)
        return [self._hit_to_result(hit, rank, "lir", query) for rank, hit in enumerate(response["hits"]["hits"], start=1)]

    def _client_lir(
        self,
        query: str,
        query_vectors: list[list[float]],
        top_k: int,
        candidate_k: int,
        filters: list[dict[str, Any]],
        vector_field: str,
    ) -> list[SearchHit]:
        candidates = rrf_fuse([self._keyword(query, candidate_k, filters), self._vector(query, candidate_k, filters)], candidate_k)
        ids = [hit.doc_id for hit in candidates]
        if not ids:
            return []
        response = self.client.mget(index=self.settings.opensearch_index, body={"ids": ids}, _source=True)
        docs_by_id = {
            str(row.get("_source", {}).get("doc_id") or row.get("_id")): row.get("_source", {})
            for row in response.get("docs", [])
            if row.get("found")
        }
        scored: list[tuple[float, SearchHit, dict[str, Any]]] = []
        for candidate in candidates:
            doc = docs_by_id.get(candidate.doc_id, candidate.doc)
            score = late_interaction_score(query_vectors, _late_vectors(doc, vector_field))
            if score <= 0:
                score = candidate.score
            scored.append((score, candidate, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        hits: list[SearchHit] = []
        for rank, (score, candidate, doc) in enumerate(scored[:top_k], start=1):
            hits.append(
                SearchHit(
                    doc_id=candidate.doc_id,
                    rank=rank,
                    score=score,
                    method="lir",
                    doc=doc,
                    excerpt=best_patch_excerpt(query, doc),
                    components={**candidate.components, "late_interaction": score},
                )
            )
        return hits

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
    if settings.require_opensearch:
        raise RuntimeError(f"OpenSearch is required but unavailable: {status.detail}")
    return LocalRetriever(settings)


def best_patch_excerpt(query: str, doc: dict[str, Any]) -> str:
    patches = doc.get("patches") or []
    if not patches:
        return excerpt_for(query, doc.get("search_text") or doc.get("body") or doc.get("summary") or "")
    query_terms = {term.lower() for term in query.split() if len(term) > 2}
    best = max(
        patches,
        key=lambda patch: sum(1 for term in query_terms if term in str(patch.get("text") or "").lower()),
    )
    label = best.get("kind") or "patch"
    if best.get("start_s") is not None and best.get("end_s") is not None:
        label = f"{label} {int(float(best['start_s']))}-{int(float(best['end_s']))}s"
    elif best.get("page") is not None:
        label = f"{label} p{best['page']}"
    return f"{label}: {excerpt_for(query, best.get('text') or '')}"


def _late_vector_field(backend: str) -> str:
    return "colbert_vectors" if backend == "colpali" else "patch_vectors"


def _late_vectors(doc: dict[str, Any], preferred_field: str) -> list[list[float]]:
    vectors = doc.get(preferred_field) or doc.get("patch_vectors") or doc.get("colbert_vectors") or []
    return vectors if isinstance(vectors, list) else []


def _is_sql_statement(query: str) -> bool:
    return query.strip().lower().startswith(("select ", "show ", "describe ", "explain "))


def _rewrite_sql_index(query: str, index_name: str, top_k: int) -> str:
    sql = query.strip().rstrip(";")
    lowered = sql.lower()
    if " from openlens " in f" {lowered} ":
        sql = sql.replace(" openlens ", f" {index_name} ")
    if " limit " not in lowered and lowered.startswith("select "):
        sql = f"{sql} LIMIT {top_k}"
    return sql
