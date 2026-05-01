from __future__ import annotations

import csv
import io
import xml.etree.ElementTree as ET
from typing import Any, Iterable

import httpx
from pypdf import PdfReader

from .data import stable_id
from .models import Asset, OpenRecord
from .text import clean_text, table_text


ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
DEFAULT_QUERY = "artemis mars earth moon hubble webb exoplanet"
NTRS_BASE_URL = "https://ntrs.nasa.gov"


class OpenSourceClient:
    def __init__(self, user_agent: str, timeout_s: float = 30):
        self.client = httpx.Client(
            timeout=timeout_s,
            follow_redirects=True,
            headers={"User-Agent": user_agent},
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "OpenSourceClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def wikimedia_images(self, query: str = DEFAULT_QUERY, limit: int = 8) -> list[OpenRecord]:
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrnamespace": "6",
            "gsrsearch": query,
            "gsrlimit": str(limit),
            "prop": "imageinfo|info",
            "inprop": "url",
            "iiprop": "url|mime|size|extmetadata",
            "iiurlwidth": "760",
            "origin": "*",
        }
        payload = self.client.get("https://commons.wikimedia.org/w/api.php", params=params).json()
        pages = payload.get("query", {}).get("pages", {})
        records: list[OpenRecord] = []
        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            metadata = info.get("extmetadata") or {}
            title = clean_text(metadata.get("ObjectName", {}).get("value") or page.get("title", ""))
            description = clean_text(metadata.get("ImageDescription", {}).get("value"), max_chars=1400)
            license_name = clean_text(metadata.get("LicenseShortName", {}).get("value"))
            license_url = clean_text(metadata.get("LicenseUrl", {}).get("value"))
            artist = clean_text(metadata.get("Artist", {}).get("value"))
            credit = clean_text(metadata.get("Credit", {}).get("value"))
            source_id = str(page.get("pageid") or title)
            records.append(
                OpenRecord(
                    doc_id="commons-" + stable_id(source_id, title),
                    source="Wikimedia Commons",
                    source_id=source_id,
                    source_url=info.get("descriptionurl") or page.get("fullurl") or "",
                    modality="image",
                    title=title.replace("File:", "", 1),
                    summary=description,
                    body=" ".join(part for part in [description, credit] if part),
                    license=license_name,
                    license_url=license_url,
                    attribution=artist,
                    tags=["image", "commons", query],
                    facets={"mime": info.get("mime"), "width": info.get("width"), "height": info.get("height")},
                    assets=[
                        Asset(
                            kind="image",
                            url=info.get("url") or "",
                            thumbnail_url=info.get("thumburl") or info.get("url") or "",
                            mime_type=info.get("mime") or "",
                            width=info.get("width"),
                            height=info.get("height"),
                        )
                    ],
                )
            )
        return records

    def arxiv_pdfs(
        self,
        query: str = DEFAULT_QUERY,
        limit: int = 8,
        fetch_pdf_text: bool = False,
        pdf_pages: int = 2,
    ) -> list[OpenRecord]:
        params = {
            "search_query": f"all:{query}",
            "start": "0",
            "max_results": str(limit),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        xml_text = self.client.get("https://export.arxiv.org/api/query", params=params).text
        root = ET.fromstring(xml_text)
        records: list[OpenRecord] = []
        for entry in root.findall("atom:entry", ARXIV_NS):
            arxiv_url = clean_text(entry.findtext("atom:id", default="", namespaces=ARXIV_NS))
            source_id = arxiv_url.rsplit("/", 1)[-1]
            title = clean_text(entry.findtext("atom:title", default="", namespaces=ARXIV_NS))
            summary = clean_text(entry.findtext("atom:summary", default="", namespaces=ARXIV_NS), max_chars=1800)
            authors = [
                clean_text(author.findtext("atom:name", default="", namespaces=ARXIV_NS))
                for author in entry.findall("atom:author", ARXIV_NS)
            ]
            categories = [node.attrib.get("term", "") for node in entry.findall("atom:category", ARXIV_NS)]
            pdf_url = ""
            for link in entry.findall("atom:link", ARXIV_NS):
                if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                    pdf_url = link.attrib.get("href", "")
                    break
            pdf_text = self._read_pdf_text(pdf_url, max_pages=pdf_pages) if fetch_pdf_text and pdf_url else ""
            records.append(
                OpenRecord(
                    doc_id="arxiv-" + stable_id(source_id),
                    source="arXiv",
                    source_id=source_id,
                    source_url=arxiv_url,
                    modality="pdf",
                    title=title,
                    summary=summary,
                    body=clean_text(summary + " " + pdf_text, max_chars=8000),
                    license="arXiv open access",
                    attribution=", ".join(author for author in authors if author),
                    published_at=clean_text(entry.findtext("atom:published", default="", namespaces=ARXIV_NS)) or None,
                    updated_at=clean_text(entry.findtext("atom:updated", default="", namespaces=ARXIV_NS)) or None,
                    tags=["pdf", "paper", *categories],
                    facets={"categories": categories, "authors": authors[:8]},
                    assets=[Asset(kind="pdf", url=pdf_url, mime_type="application/pdf")],
                )
            )
        return records

    def nasa_media(
        self,
        query: str = "artemis moon mars earth exoplanet",
        media_type: str = "image",
        limit: int = 8,
        page_size: int = 100,
    ) -> list[OpenRecord]:
        modality = {"image": "image", "video": "video", "audio": "audio"}[media_type]
        records: list[OpenRecord] = []
        page = 1
        while len(records) < limit:
            payload = self.client.get(
                "https://images-api.nasa.gov/search",
                params={
                    "q": query,
                    "media_type": media_type,
                    "page": page,
                    "page_size": min(page_size, limit - len(records)),
                },
            ).json()
            items = payload.get("collection", {}).get("items", [])
            if not items:
                break
            for item in items:
                raw_item_text = str(item).lower()
                if "archive.org" in raw_item_text or "internet archive" in raw_item_text:
                    continue
                row = (item.get("data") or [{}])[0]
                nasa_id = clean_text(row.get("nasa_id"))
                title = clean_text(row.get("title") or nasa_id)
                if not nasa_id or not title:
                    continue
                description = clean_text(row.get("description"), max_chars=2400)
                keywords = _as_list(row.get("keywords"))
                links = item.get("links") or []
                thumbnail = next((link.get("href") for link in links if link.get("href")), "")
                records.append(
                    OpenRecord(
                        doc_id="nasa-media-" + stable_id(media_type, nasa_id),
                        source="NASA Image and Video Library",
                        source_id=nasa_id,
                        source_url=f"https://images.nasa.gov/details/{nasa_id}",
                        modality=modality,  # type: ignore[arg-type]
                        title=title,
                        summary=description,
                        body=description,
                        license="NASA media guidelines",
                        license_url="https://www.nasa.gov/multimedia/guidelines/index.html",
                        attribution=clean_text(row.get("center")) or "NASA",
                        language="en",
                        published_at=clean_text(row.get("date_created")) or None,
                        tags=[modality, "nasa", media_type, *keywords[:12]],
                        facets={
                            "center": row.get("center"),
                            "media_type": media_type,
                            "keywords": keywords[:20],
                            "secondary_creator": row.get("secondary_creator"),
                        },
                        assets=[
                            Asset(
                                kind=media_type,
                                url=item.get("href") or f"https://images.nasa.gov/details/{nasa_id}",
                                thumbnail_url=thumbnail,
                                mime_type=f"{media_type}/*",
                            )
                        ],
                    )
                )
                if len(records) >= limit:
                    break
            page += 1
        return records

    def ntrs_pdfs(
        self,
        query: str = DEFAULT_QUERY,
        limit: int = 8,
        fetch_pdf_text: bool = False,
    ) -> list[OpenRecord]:
        params = {
            "q": query,
            "page.size": str(min(100, max(1, limit))),
            "distribution": "PUBLIC",
            "disseminated": "DOCUMENT_AND_METADATA",
        }
        payload = self.client.get(f"{NTRS_BASE_URL}/api/citations/search", params=params).json()
        records: list[OpenRecord] = []
        for row in payload.get("results", [])[:limit]:
            source_id = clean_text(row.get("id"))
            title = clean_text(row.get("title") or source_id)
            if not source_id or not title:
                continue
            abstract = clean_text(row.get("abstract"), max_chars=2400)
            subjects = _as_list(row.get("subjectCategories"))
            center = row.get("center") or {}
            authors = _ntrs_authors(row.get("authorAffiliations") or [])
            downloads = row.get("downloads") or []
            pdf_path = _ntrs_download_path(downloads, "pdf") or _ntrs_download_path(downloads, "original")
            fulltext_path = _ntrs_download_path(downloads, "fulltext")
            pdf_url = _ntrs_url(pdf_path)
            fulltext = self._read_text(_ntrs_url(fulltext_path), max_chars=5600) if fetch_pdf_text and fulltext_path else ""
            records.append(
                OpenRecord(
                    doc_id="ntrs-" + stable_id(source_id),
                    source="NASA STI Repository",
                    source_id=source_id,
                    source_url=f"{NTRS_BASE_URL}/citations/{source_id}",
                    modality="pdf",
                    title=title,
                    summary=abstract,
                    body=clean_text(" ".join([abstract, fulltext]), max_chars=8000),
                    license="NASA STI public record",
                    license_url="https://ntrs.nasa.gov/api/openapi/",
                    attribution=", ".join(authors) or clean_text(center.get("name")) or "NASA",
                    published_at=clean_text(row.get("distributionDate") or row.get("created")) or None,
                    tags=["pdf", "nasa", "ntrs", clean_text(row.get("stiType")), *subjects[:10]],
                    facets={
                        "center": clean_text(center.get("name")),
                        "stiType": clean_text(row.get("stiType")),
                        "stiTypeDetails": clean_text(row.get("stiTypeDetails")),
                        "subjectCategories": subjects[:12],
                        "authors": authors[:8],
                        "downloadsAvailable": row.get("downloadsAvailable"),
                    },
                    assets=[Asset(kind="pdf", url=pdf_url or f"{NTRS_BASE_URL}/citations/{source_id}", mime_type="application/pdf")],
                )
            )
        return records

    def nasa_exoplanet_rows(self, limit: int = 12) -> list[OpenRecord]:
        query = (
            "select top {limit} pl_name, hostname, disc_year, discoverymethod, pl_orbper, "
            "pl_rade, pl_bmasse, st_teff, sy_dist from pscomppars where pl_name is not null"
        ).format(limit=max(1, int(limit)))
        payload = self.client.get(
            "https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
            params={"query": query, "format": "csv"},
        ).text
        rows = list(csv.DictReader(io.StringIO(payload)))
        records: list[OpenRecord] = []
        for row in rows:
            row = {key: clean_text(value) for key, value in row.items()}
            planet = row.get("pl_name") or "unknown planet"
            host = row.get("hostname") or "unknown host"
            summary = (
                f"{planet} orbits {host}. It was discovered in {row.get('disc_year') or 'an unknown year'} "
                f"using {row.get('discoverymethod') or 'an unknown method'}."
            )
            records.append(
                OpenRecord(
                    doc_id="nasa-exoplanet-" + stable_id(planet, host),
                    source="NASA Exoplanet Archive",
                    source_id=planet,
                    source_url="https://exoplanetarchive.ipac.caltech.edu/",
                    modality="table",
                    title=f"{planet} exoplanet row",
                    summary=summary,
                    body=table_text(row),
                    license="Public domain / NASA public data",
                    attribution="NASA Exoplanet Archive",
                    tags=["table", "sql", "tap", "exoplanet", row.get("discoverymethod", "")],
                    facets={
                        "table": "pscomppars",
                        "discoverymethod": row.get("discoverymethod"),
                        "disc_year": row.get("disc_year"),
                    },
                    table=row,
                )
            )
        return records

    def _read_pdf_text(self, url: str, max_pages: int = 2) -> str:
        response = self.client.get(url)
        response.raise_for_status()
        reader = PdfReader(io.BytesIO(response.content))
        pages = []
        for page in reader.pages[:max_pages]:
            pages.append(page.extract_text() or "")
        return clean_text(" ".join(pages), max_chars=8000)

    def _read_text(self, url: str, max_chars: int = 8000) -> str:
        response = self.client.get(url)
        response.raise_for_status()
        return clean_text(response.text, max_chars=max_chars)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [clean_text(item) for item in value if clean_text(item)]
    return [clean_text(value)] if clean_text(value) else []


def _join_values(value: Any) -> str:
    return ", ".join(_as_list(value))


def _ntrs_authors(values: list[dict[str, Any]]) -> list[str]:
    authors: list[str] = []
    seen: set[str] = set()
    for value in values:
        author = ((value.get("meta") or {}).get("author") or {}).get("name")
        name = clean_text(author)
        if name and name.lower() not in seen:
            authors.append(name)
            seen.add(name.lower())
    return authors


def _ntrs_download_path(downloads: list[dict[str, Any]], key: str) -> str:
    for download in downloads:
        links = download.get("links") or {}
        path = clean_text(links.get(key))
        if path:
            return path
    return ""


def _ntrs_url(path: str) -> str:
    if not path:
        return ""
    if path.startswith("http://") or path.startswith("https://"):
        return path
    return f"{NTRS_BASE_URL}{path if path.startswith('/') else '/' + path}"


def dedupe_records(records: Iterable[OpenRecord]) -> list[OpenRecord]:
    seen: set[str] = set()
    out: list[OpenRecord] = []
    for record in records:
        if record.doc_id not in seen:
            out.append(record)
            seen.add(record.doc_id)
    return out
