from __future__ import annotations

import argparse
import gc
import time

from rich.console import Console

from openlens.config import get_settings
from openlens.qwen_embedder import QwenEmbedderError, make_embedder


def main() -> int:
    parser = argparse.ArgumentParser(description="Autotune Qwen multimodal embedding batch size on a GPU.")
    parser.add_argument("--model", default=None, help="Defaults to OPENLENS_QWEN_MODEL, normally qwen8b on H100.")
    parser.add_argument("--dimension", type=int, default=None, help="Defaults to OPENLENS_VECTOR_DIM.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--trials", type=int, default=2)
    args = parser.parse_args()

    settings = get_settings()
    console = Console()
    model = args.model or settings.qwen_model
    dimension = args.dimension or settings.vector_dim
    max_frames = args.max_frames or settings.qwen_max_frames
    fps = args.fps or settings.qwen_fps
    best: dict[str, float | int] | None = None
    batch = 1
    while batch <= args.max_batch:
        console.print(f"trying batch_size={batch} model={model} dim={dimension} max_frames={max_frames}")
        try:
            result = _bench_batch(model, dimension, batch, max_frames, fps, args.warmup, args.trials)
            console.print(
                f"  ok throughput={result['items_per_s']:.2f}/s "
                f"latency={result['latency_s']:.2f}s peak_vram={result['peak_vram_gb']:.2f}GB"
            )
            best = result
            batch *= 2
        except (RuntimeError, QwenEmbedderError) as exc:
            console.print(f"[yellow]  failed at batch_size={batch}: {type(exc).__name__}: {exc}[/yellow]")
            break
        finally:
            _cleanup_cuda()
    if not best:
        raise SystemExit("No working Qwen batch size found.")
    console.print(
        "\nRecommended H100 settings:\n"
        f"OPENLENS_EMBEDDING_BACKEND=qwen\n"
        f"OPENLENS_QWEN_MODEL={model}\n"
        f"OPENLENS_VECTOR_DIM={dimension}\n"
        f"OPENLENS_QWEN_BATCH_SIZE={best['batch_size']}\n"
        f"OPENLENS_QWEN_MAX_FRAMES={max_frames}\n"
        f"OPENLENS_QWEN_FPS={fps}"
    )
    return 0


def _bench_batch(
    model: str,
    dimension: int,
    batch_size: int,
    max_frames: int,
    fps: float,
    warmup: int,
    trials: int,
) -> dict[str, float | int]:
    embedder = make_embedder(
        "qwen",
        dimension,
        model,
        batch_size=batch_size,
        max_frames=max_frames,
        fps=fps,
    )
    texts = [
        {
            "text": (
                "Retrieve multimodal aerospace evidence about Artemis launch imagery, "
                "mission audio, exoplanet SQL table rows, and technical PDF figures."
            )
        }
        for _ in range(batch_size)
    ]
    for _ in range(warmup):
        embedder._encode_objects(texts, "Represent this document patch for multimodal retrieval.")  # noqa: SLF001
    _reset_peak_cuda()
    t0 = time.perf_counter()
    for _ in range(trials):
        embedder._encode_objects(texts, "Represent this document patch for multimodal retrieval.")  # noqa: SLF001
    latency = time.perf_counter() - t0
    return {
        "batch_size": batch_size,
        "latency_s": latency / trials,
        "items_per_s": (batch_size * trials) / latency,
        "peak_vram_gb": _peak_cuda_gb(),
    }


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
