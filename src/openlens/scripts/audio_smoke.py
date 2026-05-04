from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

from openlens.audio_embedder import ClapAudioEmbedder
from openlens.config import get_settings
from openlens.data import write_jsonl
from openlens.indexer import bulk_index, check_status, make_client, prepare_record, recreate_index
from openlens.modality_embedder import ModalityRouterEmbedder
from openlens.models import Asset, OpenRecord
from openlens.retrieval import OpenSearchRetriever


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a raw-audio CLAP smoke through OpenLens and OpenSearch.")
    parser.add_argument("--model", default=os.getenv("OPENLENS_CLAP_MODEL", "laion/clap-htsat-unfused"))
    parser.add_argument("--query", default="a sine tone audio beep")
    parser.add_argument("--index", default="openlens_audio_clap_smoke")
    parser.add_argument("--output", default="data/processed/audio_clap_smoke_embedded.jsonl")
    parser.add_argument("--skip-opensearch", action="store_true")
    parser.add_argument("--no-recreate", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    os.environ["OPENLENS_USE_REAL_AUDIO_ENCODER"] = "1"
    os.environ["OPENLENS_CLAP_MODEL"] = args.model

    audio_path = _make_smoke_wav()
    direct_embedder = ClapAudioEmbedder(model_name=args.model)
    raw_audio_vector = direct_embedder.embed_audio(str(audio_path))
    text_query_vector = direct_embedder.embed_text(args.query)

    router = ModalityRouterEmbedder()
    router._clap = direct_embedder
    record = OpenRecord(
        doc_id="audio-clap-smoke",
        source="Audio CLAP smoke",
        source_id="audio-clap-smoke",
        source_url="urn:openlens:audio-clap-smoke",
        modality="audio",
        title="Raw WAV sine tone smoke",
        summary="Generated WAV used to verify raw audio CLAP vectors in OpenSearch.",
        body="A two second 440 Hz sine tone beep rendered as raw WAV samples.",
        license="Local smoke",
        tags=["audio", "clap", "raw-wav", "sine-tone"],
        assets=[
            Asset(
                kind="audio",
                url=str(audio_path),
                mime_type="audio/wav",
                duration_s=2.0,
            )
        ],
    )
    indexed = prepare_record(record, router)
    output = Path(args.output)
    write_jsonl(output, [indexed.model_dump(mode="json")])

    search_payload: dict[str, object] = {"indexed": False}
    if not args.skip_opensearch:
        settings = replace(
            get_settings(),
            opensearch_index=args.index,
            embedding_backend="modality-router",
            vector_dim=384,
            require_opensearch=True,
        )
        status = check_status(settings)
        if not status.available:
            raise SystemExit(f"OpenSearch unavailable: {status.detail}")
        client = make_client(settings)
        if not args.no_recreate:
            recreate_index(client, args.index, settings.vector_dim)
        count = bulk_index(client, args.index, [indexed], refresh="wait_for")
        response = OpenSearchRetriever(settings).search(args.query, mode="vector", top_k=3, modality="audio")
        first = response.hits[0].to_dict() if response.hits else {}
        search_payload = {
            "indexed": True,
            "index": args.index,
            "indexed_count": count,
            "retriever": response.retriever,
            "mode": response.mode,
            "top_doc_id": first.get("doc_id", ""),
            "top_title": first.get("title", ""),
            "top_score": first.get("score", 0.0),
            "top_primary_vector_field": first.get("primary_vector_field", ""),
            "top_vector_fields": first.get("vector_fields", {}),
            "evidence": first.get("evidence", [])[:2],
        }

    payload = {
        "ok": True,
        "model": args.model,
        "raw_audio_path": str(audio_path),
        "raw_audio_vector_dim": len(raw_audio_vector),
        "text_query_vector_dim": len(text_query_vector),
        "raw_audio_text_cosine": round(float(sum(a * b for a, b in zip(raw_audio_vector, text_query_vector))), 6),
        "indexed_audio_vector_dim": len(indexed.audio_vector),
        "primary_vector_field": indexed.primary_vector_field,
        "chunk_strategy": indexed.chunk_strategy,
        "embedding_models": indexed.embedding_models,
        "patches": [patch.model_dump(mode="json") for patch in indexed.patches],
        "output": str(output),
        "opensearch": search_payload,
        "elapsed_s": round(time.perf_counter() - started, 2),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _make_smoke_wav() -> Path:
    import numpy as np
    import soundfile as sf

    path = Path("data/processed/audio_clap_smoke.wav")
    path.parent.mkdir(parents=True, exist_ok=True)
    sampling_rate = 48_000
    duration_s = 2.0
    t = np.linspace(0.0, duration_s, int(sampling_rate * duration_s), endpoint=False, dtype=np.float32)
    envelope = np.minimum(t / 0.05, 1.0) * np.minimum((duration_s - t) / 0.05, 1.0)
    waveform = (0.25 * envelope * np.sin(2.0 * np.pi * 440.0 * t)).astype("float32")
    sf.write(path, waveform, sampling_rate)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
