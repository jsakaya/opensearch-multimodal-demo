from __future__ import annotations

from dataclasses import replace

from openlens.config import Settings
from openlens.api import InlineIngestRequest, inline_request_to_record
from openlens.data import write_jsonl
from openlens.embeddings import FeatureHashEmbedder, late_interaction_score
from openlens.indexer import opensearch_source, prepare_record
from openlens.modality_embedder import ModalityRouterEmbedder
from openlens.models import Asset, OpenRecord
from openlens.qwen_embedder import qwen_runtime_status
from openlens.retrieval import LocalRetriever, SearchHit, rrf_fuse
from openlens.serverless_encoder import encode_payload
from openlens.text import compose_search_text
from openlens.video import expected_chunk_spans, is_still_frame_chunk


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


def test_patch_vectors_drive_late_interaction(tmp_path) -> None:
    settings = replace(
        Settings(),
        docs_path=tmp_path / "docs.jsonl",
        embedded_docs_path=tmp_path / "embedded.jsonl",
        vector_dim=64,
    )
    embedder = FeatureHashEmbedder(settings.vector_dim)
    positive = prepare_record(
        make_record("pdf-1", "Polar report", "Page one: glacier albedo decline. Page two: sea ice retreat.", "pdf"),
        embedder,
    )
    negative = prepare_record(
        make_record("video-1", "Archive concert clip", "Thirty seconds of jazz musicians on stage.", "video"),
        embedder,
    )
    query_vectors = embedder.embed_query_patches("glacier albedo")
    assert late_interaction_score(query_vectors, positive.patch_vectors) > late_interaction_score(
        query_vectors, negative.patch_vectors
    )
    write_jsonl(
        settings.embedded_docs_path,
        [positive.model_dump(mode="json"), negative.model_dump(mode="json")],
    )
    response = LocalRetriever(settings).search("glacier albedo", mode="lir")
    assert response.hits[0].doc_id == "pdf-1"
    assert response.hits[0].method == "lir"


def test_inline_ingest_request_becomes_stable_record() -> None:
    req = InlineIngestRequest(
        title="Fresh field note",
        body="A new observation about thermal satellite imagery.",
        modality="document",
        source="Notebook",
        tags=["thermal"],
    )
    first = inline_request_to_record(req)
    second = inline_request_to_record(req)
    assert first.doc_id == second.doc_id
    assert first.source == "Notebook"
    assert "live" in first.tags


def test_sentry_style_video_chunk_helpers() -> None:
    spans = expected_chunk_spans(65, chunk_duration_s=30, overlap_s=5)
    assert [(span.start_s, span.end_s) for span in spans] == [(0.0, 30.0), (25.0, 55.0), (50.0, 65.0)]
    assert is_still_frame_chunk([1000, 1010, 990, 1005])
    assert not is_still_frame_chunk([1000, 1400, 850, 1600])


def test_qwen_runtime_status_is_nonfatal_without_gpu() -> None:
    status = qwen_runtime_status()
    assert "torch_available" in status
    assert "cuda_available" in status
    assert "device" in status


def test_qwen_defaults_to_full_embedding_dimension(monkeypatch) -> None:
    monkeypatch.setenv("OPENLENS_EMBEDDING_BACKEND", "qwen")
    monkeypatch.delenv("OPENLENS_VECTOR_DIM", raising=False)
    assert Settings().vector_dim == 4096


def test_colpali_defaults_to_late_interaction_dimension(monkeypatch) -> None:
    monkeypatch.setenv("OPENLENS_EMBEDDING_BACKEND", "colpali")
    monkeypatch.delenv("OPENLENS_VECTOR_DIM", raising=False)
    assert Settings().vector_dim == 128


def test_modality_router_is_default_backend(monkeypatch) -> None:
    monkeypatch.delenv("OPENLENS_EMBEDDING_BACKEND", raising=False)
    monkeypatch.delenv("OPENLENS_VECTOR_DIM", raising=False)
    settings = Settings()
    assert settings.embedding_backend == "modality-router"
    assert settings.vector_dim == 384


def test_modality_router_populates_video_audio_pdf_table_fields() -> None:
    embedder = ModalityRouterEmbedder()

    video = make_record("vid-router", "Lunar rover traverse video", "A 60 second camera clip on the moon.", "video")
    video.assets = [Asset(kind="video", url="https://example.test/rover.mp4", thumbnail_url="https://example.test/rover.jpg", duration_s=60)]
    indexed_video = prepare_record(video, embedder)
    assert indexed_video.primary_vector_field == "qwen_vector"
    assert len(indexed_video.qwen_vector) == 4096
    assert indexed_video.chunk_strategy.startswith("sentrysearch-video")
    assert any(step["field"] == "qwen_vector" for step in indexed_video.encoder_plan)

    audio = make_record("aud-router", "Mission control loop", "Controllers discuss orbit insertion.", "audio")
    audio.assets = [Asset(kind="audio", url="https://example.test/loop.mp3", mime_type="audio/mpeg", duration_s=45)]
    indexed_audio = prepare_record(audio, embedder)
    assert indexed_audio.primary_vector_field == "audio_vector"
    assert len(indexed_audio.audio_vector) == 512
    assert any("CLAP" in step["model"] for step in indexed_audio.encoder_plan)

    pdf = make_record("pdf-router", "Spacesuit thermal report", "Page one has a thermal chart.", "pdf")
    pdf.assets = [Asset(kind="pdf", url="https://example.test/report.pdf", mime_type="application/pdf")]
    indexed_pdf = prepare_record(pdf, embedder)
    assert indexed_pdf.primary_vector_field == "pdf_vector"
    assert len(indexed_pdf.pdf_vector) == 128
    assert indexed_pdf.colbert_vectors
    assert indexed_pdf.patch_vector_count == len(indexed_pdf.colbert_vectors)

    table = make_record("tbl-router", "Kepler planet row", "SQL row for an exoplanet.", "table")
    table.table = {"pl_name": "Kepler-22 b", "discoverymethod": "Transit"}
    indexed_table = prepare_record(table, embedder)
    assert indexed_table.primary_vector_field == "table_vector"
    assert len(indexed_table.table_vector) == 384


def test_opensearch_source_omits_empty_modality_vectors() -> None:
    embedder = ModalityRouterEmbedder()
    indexed = prepare_record(make_record("doc-router", "Plain report", "Text-only evidence.", "document"), embedder)
    source = opensearch_source(indexed)
    assert "vector" in source
    assert "text_vector" in source
    assert "audio_vector" not in source
    assert "qwen_vector" not in source
    assert "pdf_vector" not in source


def test_index_records_expose_colbert_vectors_for_opensearch_lir() -> None:
    record = make_record("pdf-colpali", "Visual technical report", "A chart-heavy PDF page about Mars ascent.", "pdf")
    record.assets = [Asset(kind="pdf", url="https://example.test/report.pdf", mime_type="application/pdf")]
    indexed = prepare_record(record, FeatureHashEmbedder(64))
    assert indexed.colbert_vectors == indexed.patch_vectors
    assert indexed.patch_vector_count == len(indexed.patch_vectors)
    assert any(patch.asset_url.endswith("report.pdf") for patch in indexed.patches if patch.kind.startswith("pdf"))


def test_audio_records_get_transcript_style_evidence_patches() -> None:
    record = make_record("aud-1", "Apollo mission control audio", "A catalog description of launch audio.", "audio")
    record.facets["subjects"] = ["mission control", "rocket launch"]
    record.assets = [
        Asset(kind="audio-item", url="https://example.test/audio", mime_type="audio/mpeg", duration_s=90.0)
    ]
    patches = FeatureHashEmbedder(64).patch_record(record)
    kinds = {patch.kind for patch in patches}
    assert "audio_caption" in kinds
    assert "audio_transcript_or_description" in kinds
    assert any("mission control" in patch.text for patch in patches)


def test_serverless_encoder_returns_inline_feature_hash_record() -> None:
    record = make_record("srv-1", "Serverless Mars report", "Patch vectors for a Mars PDF.", "pdf")
    result = encode_payload(
        {
            "backend": "feature-hash",
            "dimension": 32,
            "records": [record.model_dump(mode="json")],
            "return_records": True,
        }
    )
    assert result["ok"] is True
    assert result["count"] == 1
    assert result["records"][0]["doc_id"] == "srv-1"
    assert result["records"][0]["colbert_vectors"]
