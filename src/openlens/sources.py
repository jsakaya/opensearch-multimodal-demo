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
DEFAULT_QUERY = "earth observation climate machine learning"


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

    def internet_archive_videos(self, query: str = DEFAULT_QUERY, limit: int = 8) -> list[OpenRecord]:
        return self.internet_archive_media("movies", "video", query=query, limit=limit)

    def internet_archive_media(
        self,
        mediatype: str,
        modality: str,
        query: str = "",
        limit: int = 8,
        page_size: int = 1000,
        sort: str = "downloads desc",
    ) -> list[OpenRecord]:
        fields = [
            "identifier",
            "title",
            "description",
            "creator",
            "licenseurl",
            "date",
            "collection",
            "subject",
            "downloads",
            "mediatype",
            "language",
        ]
        page = 1
        records: list[OpenRecord] = []
        query = clean_text(query)
        q = f"mediatype:{mediatype}"
        if query and query != "*":
            q = f"{q} AND ({query})"
        while len(records) < limit:
            params = {
                "q": q,
                "fl[]": fields,
                "rows": str(min(page_size, limit - len(records))),
                "page": str(page),
                "output": "json",
                "sort[]": sort,
            }
            payload = self.client.get("https://archive.org/advancedsearch.php", params=params).json()
            docs = payload.get("response", {}).get("docs", [])
            if not docs:
                break
            for row in docs:
                identifier = clean_text(row.get("identifier"))
                if not identifier:
                    continue
                records.append(self._internet_archive_record(row, identifier, modality))
                if len(records) >= limit:
                    break
            page += 1
        return records

    def _internet_archive_record(self, row: dict[str, Any], identifier: str, modality: str) -> OpenRecord:
        title = clean_text(_join_values(row.get("title")) or identifier)
        description = clean_text(_join_values(row.get("description")), max_chars=2400)
        creator = _join_values(row.get("creator"))
        collections = _as_list(row.get("collection"))
        subjects = _as_list(row.get("subject"))
        license_url = clean_text(_join_values(row.get("licenseurl")))
        mediatype = clean_text(row.get("mediatype"))
        thumbnail_url = f"https://archive.org/services/img/{identifier}"
        asset_kind = "thumbnail"
        mime_type = "image/jpeg"
        if modality == "audio":
            asset_kind = "audio-item"
            mime_type = "audio/*"
        elif modality == "video":
            asset_kind = "video-item"
            mime_type = "video/*"
        elif modality == "pdf":
            asset_kind = "text-item"
            mime_type = "application/pdf"
        return OpenRecord(
            doc_id="archive-" + stable_id(identifier),
            source="Internet Archive",
            source_id=identifier,
            source_url=f"https://archive.org/details/{identifier}",
            modality=modality,  # type: ignore[arg-type]
            title=title,
            summary=description,
            body=description,
            license=license_url or "Internet Archive metadata",
            license_url=license_url,
            attribution=creator,
            language=_join_values(row.get("language")) or "unknown",
            published_at=clean_text(row.get("date")) or None,
            tags=[modality, "internet archive", mediatype, *collections[:8], *subjects[:8]],
            facets={
                "collection": collections,
                "subjects": subjects[:16],
                "downloads": row.get("downloads"),
                "mediatype": mediatype,
            },
            assets=[
                Asset(
                    kind=asset_kind,
                    url=f"https://archive.org/details/{identifier}",
                    thumbnail_url=thumbnail_url,
                    mime_type=mime_type,
                )
            ],
        )

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


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [clean_text(item) for item in value if clean_text(item)]
    return [clean_text(value)] if clean_text(value) else []


def _join_values(value: Any) -> str:
    return ", ".join(_as_list(value))


def dedupe_records(records: Iterable[OpenRecord]) -> list[OpenRecord]:
    seen: set[str] = set()
    out: list[OpenRecord] = []
    for record in records:
        if record.doc_id not in seen:
            out.append(record)
            seen.add(record.doc_id)
    return out
