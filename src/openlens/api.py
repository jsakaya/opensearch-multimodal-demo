from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import get_settings
from .data import append_or_replace_jsonl, stable_id
from .embeddings import FeatureHashEmbedder
from .indexer import bulk_index, check_status, make_client, prepare_record, recreate_index
from .models import Modality, OpenRecord
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


def inline_request_to_record(req: InlineIngestRequest) -> OpenRecord:
    title = clean_text(req.title)
    body = clean_text(req.body)
    source = clean_text(req.source) or "Live ingest"
    source_url = clean_text(req.source_url)
    doc_id = "live-" + stable_id(source, source_url, title, body)
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
    }


@app.get("/api/examples")
def examples() -> dict[str, Any]:
    return {
        "queries": [
            "satellite imagery climate change",
            "PDF papers about machine learning for earth observation",
            "public domain videos of scientific archives",
            "transit method exoplanet rows",
            "glacier retreat images",
        ],
        "modes": ["hybrid", "keyword", "vector"],
        "modalities": ["image", "pdf", "document", "video", "table"],
    }


@app.get("/api/search")
def search(
    q: str = Query(..., min_length=2),
    mode: SearchMode = "hybrid",
    top_k: int = Query(12, ge=1, le=50),
    candidate_k: int = Query(80, ge=10, le=500),
    modality: str | None = None,
    source: str | None = None,
    prefer_opensearch: bool = True,
) -> dict[str, Any]:
    retriever = cached_retriever(prefer_opensearch)
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
    embedder = FeatureHashEmbedder(settings.vector_dim)
    indexed = prepare_record(record, embedder)

    append_or_replace_jsonl(settings.docs_path, [record.model_dump(mode="json")])
    append_or_replace_jsonl(settings.embedded_docs_path, [indexed.model_dump(mode="json")])

    os_status = check_status(settings)
    indexed_to = "local-jsonl"
    if prefer_opensearch and os_status.available:
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
