from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoChunk:
    ordinal: int
    start_s: float
    end_s: float


def expected_chunk_spans(duration_s: float, chunk_duration_s: float = 30.0, overlap_s: float = 5.0) -> list[VideoChunk]:
    duration_s = max(0.0, float(duration_s or 0.0))
    chunk_duration_s = max(1.0, float(chunk_duration_s))
    overlap_s = min(max(0.0, float(overlap_s)), chunk_duration_s - 0.1)
    step = chunk_duration_s - overlap_s
    spans: list[VideoChunk] = []
    start = 0.0
    ordinal = 0
    while start < duration_s or (duration_s == 0.0 and not spans):
        end = min(duration_s, start + chunk_duration_s) if duration_s else chunk_duration_s
        spans.append(VideoChunk(ordinal=ordinal, start_s=round(start, 3), end_s=round(end, 3)))
        ordinal += 1
        if end >= duration_s and duration_s:
            break
        start += step
    return spans


def is_still_frame_chunk(sample_jpeg_sizes: list[int], tolerance: float = 0.08) -> bool:
    """Heuristic inspired by SentrySearch: static clips compress to nearly identical frame sizes."""

    sizes = [int(size) for size in sample_jpeg_sizes if int(size) > 0]
    if len(sizes) < 3:
        return False
    mean_size = sum(sizes) / len(sizes)
    if mean_size <= 0:
        return False
    spread = max(sizes) - min(sizes)
    return (spread / mean_size) <= tolerance
