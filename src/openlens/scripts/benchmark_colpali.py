from __future__ import annotations

import argparse
import gc
import time

from rich.console import Console

from openlens.colpali_embedder import ColPaliEmbedder, ColPaliEmbedderError
from openlens.config import get_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Autotune ColPali visual-document embedding batch size on a GPU.")
    parser.add_argument("--model", default=None, help="Defaults to OPENLENS_COLPALI_MODEL.")
    parser.add_argument("--dimension", type=int, default=None, help="Defaults to OPENLENS_VECTOR_DIM.")
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--max-patch-vectors", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--trials", type=int, default=2)
    args = parser.parse_args()

    settings = get_settings()
    console = Console()
    model = args.model or settings.colpali_model
    dimension = args.dimension or settings.vector_dim
    max_patch_vectors = args.max_patch_vectors or settings.colpali_max_patch_vectors
    best: dict[str, float | int] | None = None
    batch = 1
    while batch <= args.max_batch:
        console.print(f"trying batch_size={batch} model={model} dim={dimension} max_patch_vectors={max_patch_vectors}")
        try:
            result = _bench_batch(model, dimension, batch, max_patch_vectors, args.warmup, args.trials)
            console.print(
                f"  ok throughput={result['pages_per_s']:.2f} pages/s "
                f"latency={result['latency_s']:.2f}s peak_vram={result['peak_vram_gb']:.2f}GB "
                f"vectors/page={result['vectors_per_page']}"
            )
            best = result
            batch *= 2
        except (RuntimeError, ColPaliEmbedderError) as exc:
            console.print(f"[yellow]  failed at batch_size={batch}: {type(exc).__name__}: {exc}[/yellow]")
            break
        finally:
            _cleanup_cuda()
    if not best:
        raise SystemExit("No working ColPali batch size found.")
    console.print(
        "\nRecommended H100/H200 settings:\n"
        "OPENLENS_EMBEDDING_BACKEND=colpali\n"
        f"OPENLENS_COLPALI_MODEL={model}\n"
        f"OPENLENS_VECTOR_DIM={dimension}\n"
        f"OPENLENS_COLPALI_BATCH_SIZE={best['batch_size']}\n"
        f"OPENLENS_COLPALI_MAX_PATCH_VECTORS={max_patch_vectors}"
    )
    return 0


def _bench_batch(
    model: str,
    dimension: int,
    batch_size: int,
    max_patch_vectors: int,
    warmup: int,
    trials: int,
) -> dict[str, float | int]:
    embedder = ColPaliEmbedder(
        model_name=model,
        dimension=dimension,
        batch_size=batch_size,
        max_patch_vectors=max_patch_vectors,
    )
    images = [_demo_page_image(idx) for idx in range(batch_size)]
    for _ in range(warmup):
        embedder._encode_images(images)  # noqa: SLF001
    _reset_peak_cuda()
    t0 = time.perf_counter()
    vector_count = 0
    for _ in range(trials):
        vectors = embedder._encode_images(images)  # noqa: SLF001
        vector_count = sum(len(item) for item in vectors)
    latency = time.perf_counter() - t0
    return {
        "batch_size": batch_size,
        "latency_s": latency / trials,
        "pages_per_s": (batch_size * trials) / latency,
        "peak_vram_gb": _peak_cuda_gb(),
        "vectors_per_page": int(vector_count / batch_size) if batch_size else 0,
    }


def _demo_page_image(idx: int):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (896, 1152), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 80, 836, 220), outline="black", width=4)
    draw.text((90, 125), f"NASA TECHNICAL REPORT PAGE {idx + 1}", fill="black")
    draw.rectangle((80, 290, 420, 780), outline="black", width=3)
    draw.text((120, 500), "PROPULSION\nDIAGRAM", fill="black")
    for row in range(6):
        y = 300 + row * 62
        draw.line((480, y, 820, y), fill="black", width=2)
        draw.text((500, y + 18), f"mission metric {row + 1}: {idx + row + 42}", fill="black")
    draw.line((90, 980, 810, 840), fill="black", width=5)
    return image


def _reset_peak_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _peak_cuda_gb() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / 1024**3)
    except Exception:
        pass
    return 0.0


def _cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
