from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import get_settings
from .data import append_or_replace_jsonl, stable_id
from .indexer import bulk_index, check_status, make_client, prepare_record, recreate_index
from .models import Asset, Modality, OpenRecord
from .qwen_embedder import QwenEmbedderError, make_embedder, qwen_runtime_status
from .retrieval import LocalRetriever, OpenSearchRetriever, SearchMode, make_retriever
from .text import clean_text


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"


class InlineIngestRequest(BaseModel):
    title: str = Field(min_length=2)
    body: str = Field(min_length=2)
    modality: Modality = "document"
    summary: str = ""
    source: str = "Live ingest"
    source_url: str = ""
    license: str = "User supplied"
    license_url: str = ""
    attribution: str = ""
    tags: list[str] = Field(default_factory=list)
    facets: dict[str, Any] = Field(default_factory=dict)
    asset_url: str = ""
    thumbnail_url: str = ""
    duration_s: float | None = None


def inline_request_to_record(req: InlineIngestRequest) -> OpenRecord:
    title = clean_text(req.title)
    body = clean_text(req.body)
    source = clean_text(req.source) or "Live ingest"
    source_url = clean_text(req.source_url)
    doc_id = "live-" + stable_id(source, source_url, title, body)
    asset_url = clean_text(req.asset_url)
    assets = []
    if asset_url:
        assets.append(
            Asset(
                kind=req.modality,
                url=asset_url,
                thumbnail_url=clean_text(req.thumbnail_url) or asset_url,
                duration_s=req.duration_s,
            )
        )
    return OpenRecord(
        doc_id=doc_id,
        source=source,
        source_id=doc_id,
        source_url=source_url or f"urn:openlens:{doc_id}",
        modality=req.modality,
        title=title,
        summary=clean_text(req.summary) or clean_text(body, max_chars=240),
        body=body,
        license=clean_text(req.license),
        license_url=clean_text(req.license_url),
        attribution=clean_text(req.attribution),
        tags=[req.modality, "live", *req.tags],
        facets=req.facets,
        assets=assets,
    )


@lru_cache(maxsize=2)
def cached_retriever(prefer_opensearch: bool = True) -> LocalRetriever | OpenSearchRetriever:
    return make_retriever(get_settings(), prefer_opensearch=prefer_opensearch)


def reset_retriever_cache() -> None:
    cached_retriever.cache_clear()


app = FastAPI(title="OpenLens OpenSearch Demo")
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    path = STATIC_DIR / "index.html"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "<!doctype html><title>OpenLens</title><body>OpenLens API is running.</body>"


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/status")
def status() -> dict[str, Any]:
    settings = get_settings()
    os_status = check_status(settings)
    return {
        "opensearch": os_status.__dict__,
        "index": settings.opensearch_index,
        "docs_path": str(settings.docs_path),
        "embedded_docs_path": str(settings.embedded_docs_path),
        "local_docs_available": settings.docs_path.exists() or settings.embedded_docs_path.exists(),
        "vector_dim": settings.vector_dim,
        "embedding_backend": settings.embedding_backend,
        "qwen_model": settings.qwen_model,
        "qwen_batch_size": settings.qwen_batch_size,
        "qwen_max_frames": settings.qwen_max_frames,
        "qwen_fps": settings.qwen_fps,
        "qwen_runtime": qwen_runtime_status(),
        "require_opensearch": settings.require_opensearch,
    }


@app.post("/api/prewarm")
def prewarm() -> dict[str, Any]:
    settings = get_settings()
    started = time.perf_counter()
    try:
        retriever = cached_retriever(True)
        query_vectors = retriever.embedder.embed_query_patches(
            "warm up Qwen multimodal retrieval for OpenSearch hybrid vector LIR search"
        )
    except (RuntimeError, QwenEmbedderError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    vector_dim = len(query_vectors[0]) if query_vectors else 0
    return {
        "status": "ready",
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "retriever": type(retriever).__name__,
        "embedding_backend": settings.embedding_backend,
        "embedding_model": settings.qwen_model if settings.embedding_backend == "qwen" else "feature-hash",
        "vector_dim": vector_dim,
        "query_vectors": len(query_vectors),
        "qwen_batch_size": settings.qwen_batch_size,
        "qwen_max_frames": settings.qwen_max_frames,
        "qwen_runtime": qwen_runtime_status(),
    }


@app.get("/api/examples")
def examples() -> dict[str, Any]:
    return {
        "queries": [
            "Artemis moon landing mission video",
            "Mars rover helicopter image",
            "NASA technical reports about exoplanet occurrence rates",
            "mission control audio schedule inventory",
            "transit method exoplanet rows",
            "SELECT title, body FROM openlens WHERE modality = 'table' LIMIT 5",
            "Hubble Webb dark matter galaxy images",
        ],
        "modes": ["hybrid", "keyword", "vector", "lir", "sql"],
        "modalities": ["image", "pdf", "document", "video", "audio", "table"],
    }


@app.get("/api/search")
def search(
    q: str = Query(..., min_length=2),
    mode: SearchMode = "hybrid",
    top_k: int = Query(12, ge=1, le=50),
    candidate_k: int = Query(80, ge=10, le=500),
    modality: str | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    try:
        retriever = cached_retriever(True)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return retriever.search(
        q,
        mode=mode,
        top_k=top_k,
        candidate_k=candidate_k,
        modality=modality or None,
        source=source or None,
    ).to_dict()


@app.post("/api/ingest")
def ingest(req: InlineIngestRequest, prefer_opensearch: bool = True) -> dict[str, Any]:
    settings = get_settings()
    record = inline_request_to_record(req)
    embedder = make_embedder(
        settings.embedding_backend,
        settings.vector_dim,
        settings.qwen_model,
        batch_size=settings.qwen_batch_size,
        max_frames=settings.qwen_max_frames,
        fps=settings.qwen_fps,
    )
    indexed = prepare_record(record, embedder)

    append_or_replace_jsonl(settings.docs_path, [record.model_dump(mode="json")])
    append_or_replace_jsonl(settings.embedded_docs_path, [indexed.model_dump(mode="json")])

    os_status = check_status(settings)
    if not os_status.available:
        raise HTTPException(status_code=503, detail=f"OpenSearch is required for ingest: {os_status.detail}")
    client = make_client(settings)
    if not client.indices.exists(index=settings.opensearch_index):
        recreate_index(client, settings.opensearch_index, settings.vector_dim)
    bulk_index(client, settings.opensearch_index, [indexed], refresh="wait_for")
    indexed_to = "opensearch"
    os_status = check_status(settings)

    reset_retriever_cache()
    return {
        "record": indexed.model_dump(mode="json", exclude={"vector"}),
        "indexed_to": indexed_to,
        "opensearch": os_status.__dict__,
    }


def main() -> None:
    import uvicorn

    uvicorn.run("openlens.api:app", host="0.0.0.0", port=8787, reload=True)


if __name__ == "__main__":
    main()
