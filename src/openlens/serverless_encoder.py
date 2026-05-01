from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

from .colpali_embedder import colpali_runtime_status
from .indexer import prepare_records
from .models import OpenRecord
from .qwen_embedder import make_embedder, qwen_runtime_status


_EMBEDDER_CACHE: dict[tuple[Any, ...], Any] = {}


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _default_dimension(backend: str) -> int:
    if os.getenv("OPENLENS_VECTOR_DIM"):
        return int(os.environ["OPENLENS_VECTOR_DIM"])
    if backend in {"modality-router", "modality", "routed"}:
        return 384
    if backend == "qwen":
        return 4096
    if backend == "colpali":
        return 128
    return 384


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_records_blob(blob: str) -> list[dict[str, Any]]:
    text = blob.strip()
    if not text:
        return []
    if text[0] in "[{":
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            parsed = parsed.get("records", [parsed])
        if not isinstance(parsed, list):
            raise ValueError("JSON records payload must be a list, object, or {'records': [...]}")
        return [dict(row) for row in parsed]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _load_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if payload.get("record"):
        rows.append(dict(payload["record"]))
    if payload.get("records"):
        rows.extend(dict(row) for row in payload["records"])

    records_url = str(payload.get("records_url") or "")
    if records_url:
        if records_url.startswith(("http://", "https://")):
            response = httpx.get(records_url, timeout=httpx.Timeout(None, connect=10.0), follow_redirects=True)
            response.raise_for_status()
            rows.extend(_parse_records_blob(response.text))
        else:
            path = Path(records_url.removeprefix("file://")).expanduser()
            rows.extend(_parse_records_blob(path.read_text(encoding="utf-8")))
    return rows


def _write_output(url: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    body = "\n".join(json.dumps(row, ensure_ascii=True, sort_keys=True) for row in rows) + "\n"
    if url.startswith(("http://", "https://")):
        response = httpx.put(
            url,
            content=body.encode("utf-8"),
            headers={"Content-Type": "application/x-ndjson"},
            timeout=httpx.Timeout(None, connect=10.0),
        )
        response.raise_for_status()
        return {"output_url": url, "output_status": response.status_code, "output_bytes": len(body.encode("utf-8"))}

    path = Path(url.removeprefix("file://")).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return {"output_url": str(path), "output_bytes": len(body.encode("utf-8"))}


def _embedder_key(payload: dict[str, Any]) -> tuple[Any, ...]:
    backend = str(payload.get("backend") or os.getenv("OPENLENS_EMBEDDING_BACKEND", "colpali"))
    dimension = int(payload.get("dimension") or _default_dimension(backend))
    return (
        backend,
        dimension,
        str(payload.get("qwen_model") or os.getenv("OPENLENS_QWEN_MODEL", "qwen8b")),
        int(payload.get("qwen_batch_size") or _env_int("OPENLENS_QWEN_BATCH_SIZE", 1)),
        int(payload.get("qwen_max_frames") or _env_int("OPENLENS_QWEN_MAX_FRAMES", 32)),
        float(payload.get("qwen_fps") or _env_float("OPENLENS_QWEN_FPS", 1.0)),
        str(payload.get("colpali_model") or os.getenv("OPENLENS_COLPALI_MODEL", "colpali-v1.3")),
        int(payload.get("colpali_batch_size") or _env_int("OPENLENS_COLPALI_BATCH_SIZE", 2)),
        int(payload.get("colpali_max_pages") or _env_int("OPENLENS_COLPALI_MAX_PAGES", 1)),
        int(payload.get("colpali_max_patch_vectors") or _env_int("OPENLENS_COLPALI_MAX_PATCH_VECTORS", 1024)),
        float(payload.get("colpali_image_timeout_s") or _env_float("OPENLENS_COLPALI_IMAGE_TIMEOUT_S", 20.0)),
    )


def _get_embedder(payload: dict[str, Any]) -> Any:
    key = _embedder_key(payload)
    if key not in _EMBEDDER_CACHE:
        (
            backend,
            dimension,
            qwen_model,
            qwen_batch_size,
            qwen_max_frames,
            qwen_fps,
            colpali_model,
            colpali_batch_size,
            colpali_max_pages,
            colpali_max_patch_vectors,
            colpali_image_timeout_s,
        ) = key
        _EMBEDDER_CACHE[key] = make_embedder(
            backend=str(backend),
            dimension=int(dimension),
            model_name=str(qwen_model),
            batch_size=int(qwen_batch_size),
            max_frames=int(qwen_max_frames),
            fps=float(qwen_fps),
            colpali_model=str(colpali_model),
            colpali_batch_size=int(colpali_batch_size),
            colpali_max_pages=int(colpali_max_pages),
            colpali_max_patch_vectors=int(colpali_max_patch_vectors),
            colpali_image_timeout_s=float(colpali_image_timeout_s),
        )
    return _EMBEDDER_CACHE[key]


def runtime_status() -> dict[str, Any]:
    return {
        "ok": True,
        "backend_default": os.getenv("OPENLENS_EMBEDDING_BACKEND", "modality-router"),
        "colpali": colpali_runtime_status(),
        "qwen": qwen_runtime_status(),
    }


def encode_payload(payload: dict[str, Any]) -> dict[str, Any]:
    action = str(payload.get("action") or "encode")
    if action in {"health", "status"}:
        return runtime_status()

    rows = _load_records(payload)
    if not rows:
        raise ValueError("input must include record, records, or records_url")

    max_records = int(payload.get("max_records") or _env_int("OPENLENS_SERVERLESS_MAX_RECORDS", 10000))
    if len(rows) > max_records:
        raise ValueError(f"received {len(rows)} records, max_records is {max_records}")

    inline_max = int(payload.get("inline_max_records") or _env_int("OPENLENS_SERVERLESS_INLINE_MAX_RECORDS", 16))
    output_url = str(payload.get("output_url") or "")
    if "return_records" in payload:
        return_records = _bool(payload.get("return_records"))
    else:
        return_records = not output_url and len(rows) <= inline_max

    if return_records and len(rows) > inline_max and not _bool(payload.get("allow_large_inline")):
        raise ValueError(
            f"inline response requested for {len(rows)} records; use output_url or set allow_large_inline=true"
        )
    if not return_records and not output_url:
        raise ValueError("set output_url or return_records=true so encoded records are not discarded")

    embedder = _get_embedder(payload)
    records = [OpenRecord.model_validate(row) for row in rows]
    started = time.monotonic()
    indexed = prepare_records(records, embedder)
    elapsed_s = max(time.monotonic() - started, 1e-9)
    encoded_rows = [record.model_dump(mode="json") for record in indexed]

    result: dict[str, Any] = {
        "ok": True,
        "count": len(indexed),
        "embedding_backend": getattr(embedder, "backend", "unknown"),
        "embedding_model": getattr(embedder, "model_name", "unknown"),
        "dimension": getattr(embedder, "dimension", None),
        "elapsed_s": round(elapsed_s, 4),
        "records_per_s": round(len(indexed) / elapsed_s, 4),
        "patch_vectors": sum(record.patch_vector_count for record in indexed),
        "patch_vectors_per_record": round(
            sum(record.patch_vector_count for record in indexed) / max(len(indexed), 1),
            2,
        ),
    }
    if output_url:
        result.update(_write_output(output_url, encoded_rows))
    if return_records:
        result["records"] = encoded_rows
    return result


def handle_event(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("input") if isinstance(event.get("input"), dict) else event
    try:
        return encode_payload(dict(payload or {}))
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
