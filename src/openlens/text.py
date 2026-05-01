from __future__ import annotations

import html
import re
from typing import Any

from .models import OpenRecord


TAG_RE = re.compile(r"<[^>]+>")
SPACE_RE = re.compile(r"\s+")


def clean_text(value: Any, max_chars: int | None = None) -> str:
    text = html.unescape(str(value or ""))
    text = TAG_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    if max_chars and len(text) > max_chars:
        return text[:max_chars].rsplit(" ", 1)[0].rstrip() + "..."
    return text


def table_text(row: dict[str, Any]) -> str:
    parts = []
    for key, value in row.items():
        if value is None or value == "":
            continue
        parts.append(f"{key.replace('_', ' ')}: {clean_text(value)}")
    return "; ".join(parts)


def compose_search_text(record: OpenRecord | dict[str, Any]) -> str:
    if isinstance(record, dict):
        record = OpenRecord.model_validate(record)
    asset_text = " ".join(
        clean_text(f"{asset.kind} {asset.mime_type} {asset.url}") for asset in record.assets if asset.url
    )
    facet_text = table_text(record.facets)
    row_text = table_text(record.table)
    tags = " ".join(record.tags)
    return clean_text(
        " ".join(
            [
                record.modality,
                record.source,
                record.title,
                record.summary,
                record.body,
                tags,
                facet_text,
                row_text,
                asset_text,
                record.license,
                record.attribution,
            ]
        )
    )


def excerpt_for(query: str, text: str, width: int = 360) -> str:
    text = clean_text(text)
    if len(text) <= width:
        return text
    tokens = [re.escape(token.lower()) for token in query.split() if len(token) > 2]
    if tokens:
        match = re.search("|".join(tokens), text.lower())
        if match:
            start = max(0, match.start() - width // 3)
            end = min(len(text), start + width)
            prefix = "..." if start else ""
            suffix = "..." if end < len(text) else ""
            return prefix + text[start:end].strip() + suffix
    return text[:width].rstrip() + "..."
