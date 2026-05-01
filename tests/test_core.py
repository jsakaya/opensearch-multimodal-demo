from __future__ import annotations

from dataclasses import replace

from openlens.config import Settings
from openlens.data import write_jsonl
from openlens.embeddings import FeatureHashEmbedder
from openlens.indexer import prepare_record
from openlens.models import OpenRecord
from openlens.retrieval import LocalRetriever, SearchHit, rrf_fuse
from openlens.text import compose_search_text


def make_record(doc_id: str, title: str, body: str, modality: str = "document") -> OpenRecord:
    return OpenRecord(
        doc_id=doc_id,
        source="fixture",
        source_id=doc_id,
        source_url=f"https://example.test/{doc_id}",
        modality=modality,  # type: ignore[arg-type]
        title=title,
        summary=body[:80],
        body=body,
        license="CC0",
        tags=[modality, "fixture"],
    )


def test_search_text_preserves_multimodal_fields() -> None:
    record = make_record(
        "img-1",
        "Landsat image of glacier retreat",
        "A satellite image caption about ice loss and climate records.",
        modality="image",
    )
    text = compose_search_text(record)
    assert "image" in text
    assert "glacier retreat" in text
    assert "satellite image caption" in text


def test_hash_embedder_scores_related_text_higher() -> None:
    embedder = FeatureHashEmbedder(dimension=64)
    query = embedder.embed_text("glacier satellite image")
    positive = embedder.embed_record(
        make_record("a", "Glacier retreat from orbit", "Satellite image evidence of melting ice.", "image")
    )
    negative = embedder.embed_record(make_record("b", "Jazz invoice table", "SQL row about albums and invoices.", "table"))
    assert sum(a * b for a, b in zip(query, positive)) > sum(a * b for a, b in zip(query, negative))


def test_rrf_fusion_promotes_cross_mode_agreement() -> None:
    doc_a = {"doc_id": "a", "title": "a"}
    doc_b = {"doc_id": "b", "title": "b"}
    fused = rrf_fuse(
        [
            [SearchHit("a", 1, 10, "keyword", doc_a, "", {}), SearchHit("b", 2, 8, "keyword", doc_b, "", {})],
            [SearchHit("b", 1, 9, "vector", doc_b, "", {})],
        ],
        top_k=2,
    )
    assert fused[0].doc_id == "b"
    assert fused[0].method == "hybrid"
    assert set(fused[0].components) == {"keyword", "vector"}


def test_local_retriever_filters_modalities(tmp_path) -> None:
    settings = replace(
        Settings(),
        docs_path=tmp_path / "docs.jsonl",
        embedded_docs_path=tmp_path / "embedded.jsonl",
        vector_dim=64,
    )
    embedder = FeatureHashEmbedder(settings.vector_dim)
    rows = [
        prepare_record(
            make_record("image-1", "Satellite image of wildfire smoke", "Smoke plume visible from orbit.", "image"),
            embedder,
        ).model_dump(mode="json"),
        prepare_record(
            make_record("table-1", "Exoplanet discovery row", "Transit method planet row in a SQL table.", "table"),
            embedder,
        ).model_dump(mode="json"),
    ]
    write_jsonl(settings.embedded_docs_path, rows)
    retriever = LocalRetriever(settings)
    response = retriever.search("satellite smoke", modality="image")
    assert response.hits
    assert {hit.doc["modality"] for hit in response.hits} == {"image"}
