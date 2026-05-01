from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from .models import OpenRecord


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def stable_id(*parts: object) -> str:
    raw = "\x1f".join(str(part or "") for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_records(path: Path) -> list[OpenRecord]:
    return [OpenRecord.model_validate(row) for row in read_jsonl(path)]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def append_or_replace_jsonl(path: Path, rows: Iterable[dict[str, Any]], key: str = "doc_id") -> None:
    merged = {str(row[key]): row for row in read_jsonl(path) if key in row}
    for row in rows:
        merged[str(row[key])] = row
    write_jsonl(path, merged.values())
