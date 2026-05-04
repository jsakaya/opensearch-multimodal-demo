from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from openlens.data import write_jsonl
from openlens.indexer import prepare_record
from openlens.mlx_embedder import MlxQwenVlEmbedder, MlxTextEmbedder, mlx_runtime_status
from openlens.models import Asset, OpenRecord


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a tiny Apple MLX embedding smoke for OpenLens.")
    parser.add_argument("--text-model", default="mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ")
    parser.add_argument("--qwen-vl-model", default="mlx-community/Qwen3-VL-Embedding-2B-6bit")
    parser.add_argument("--skip-qwen-vl", action="store_true")
    parser.add_argument("--output", default="data/processed/mlx_smoke_embedded.jsonl")
    args = parser.parse_args()

    started = time.perf_counter()
    text_embedder = MlxTextEmbedder(model_name=args.text_model, dimension=1024)
    text_vector = text_embedder.embed_text("Artemis moon landing mission video")
    rows = [
        prepare_record(
            OpenRecord(
                doc_id="mlx-smoke-text",
                source="MLX smoke",
                source_id="mlx-smoke-text",
                source_url="urn:openlens:mlx-smoke-text",
                modality="document",
                title="MLX text embedding smoke",
                summary="Apple GPU Qwen3 text embedding smoke for OpenLens.",
                body="Artemis moon landing mission video and mission control audio schedule evidence.",
                license="Local smoke",
                tags=["mlx", "apple-gpu", "qwen3"],
            ),
            text_embedder,
        ).model_dump(mode="json")
    ]

    qwen_vl_shape: list[int] = []
    if not args.skip_qwen_vl:
        image_path = _make_smoke_image()
        qwen_vl = MlxQwenVlEmbedder(model_name=args.qwen_vl_model, dimension=2048)
        image_record = OpenRecord(
            doc_id="mlx-smoke-image",
            source="MLX smoke",
            source_id="mlx-smoke-image",
            source_url="urn:openlens:mlx-smoke-image",
            modality="image",
            title="White moon disk on black space",
            summary="Generated bitmap used to verify Qwen3-VL MLX image-text embedding.",
            body="A white moon-like circle on a black background.",
            license="Local smoke",
            tags=["mlx", "qwen3-vl", "image"],
            assets=[Asset(kind="image", url=str(image_path), thumbnail_url=str(image_path), mime_type="image/png")],
        )
        indexed_image = prepare_record(image_record, qwen_vl)
        rows.append(indexed_image.model_dump(mode="json"))
        qwen_vl_shape = [len(indexed_image.vector)]

    output = Path(args.output)
    write_jsonl(output, rows)
    payload = {
        "ok": True,
        "runtime": mlx_runtime_status(),
        "text_model": args.text_model,
        "text_vector_dim": len(text_vector),
        "qwen_vl_model": "" if args.skip_qwen_vl else args.qwen_vl_model,
        "qwen_vl_vector_dim": qwen_vl_shape[0] if qwen_vl_shape else 0,
        "records": len(rows),
        "output": str(output),
        "elapsed_s": round(time.perf_counter() - started, 2),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _make_smoke_image() -> Path:
    from PIL import Image, ImageDraw

    path = Path("data/processed/mlx_smoke_moon.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (224, 224), "black")
    draw = ImageDraw.Draw(image)
    draw.ellipse((52, 52, 172, 172), fill="white")
    draw.text((52, 184), "moon", fill="white")
    image.save(path)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
