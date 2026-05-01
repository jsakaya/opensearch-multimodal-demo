from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


Modality = Literal["image", "pdf", "document", "video", "table", "mixed"]


class Asset(BaseModel):
    kind: str
    url: str
    thumbnail_url: str = ""
    mime_type: str = ""
    width: int | None = None
    height: int | None = None
    duration_s: float | None = None


class OpenRecord(BaseModel):
    doc_id: str
    source: str
    source_id: str
    source_url: str
    modality: Modality
    title: str
    summary: str = ""
    body: str = ""
    license: str = ""
    license_url: str = ""
    attribution: str = ""
    language: str = "en"
    published_at: str | None = None
    updated_at: str | None = None
    tags: list[str] = Field(default_factory=list)
    facets: dict[str, Any] = Field(default_factory=dict)
    table: dict[str, Any] = Field(default_factory=dict)
    assets: list[Asset] = Field(default_factory=list)

    @field_validator("doc_id", "source", "source_id", "title")
    @classmethod
    def require_text(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("required text field is empty")
        return value

    @field_validator("tags")
    @classmethod
    def clean_tags(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for value in values:
            tag = str(value or "").strip().lower()
            if tag and tag not in seen:
                cleaned.append(tag)
                seen.add(tag)
        return cleaned


class IndexedRecord(OpenRecord):
    search_text: str
    vector: list[float]
    indexed_at: str
