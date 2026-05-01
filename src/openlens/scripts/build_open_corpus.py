from __future__ import annotations

import argparse

from rich.console import Console

from openlens.config import get_settings
from openlens.data import write_jsonl
from openlens.sources import DEFAULT_QUERY, OpenSourceClient, dedupe_records


BULK_IA_SOURCES = {
    "ia-texts": ("texts", "pdf"),
    "ia-images": ("image", "image"),
    "ia-videos": ("movies", "video"),
    "ia-audio": ("audio", "audio"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an open multimodal corpus from public sources.")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--limit-per-source", type=int, default=8)
    parser.add_argument("--target-docs", type=int, default=0, help="Target record count for bulk Internet Archive builds.")
    parser.add_argument("--bulk-internet-archive", action="store_true", help="Page Internet Archive until target-docs is met.")
    parser.add_argument("--ia-query", default="*", help="Optional Internet Archive query term for bulk builds.")
    parser.add_argument("--ia-page-size", type=int, default=1000)
    parser.add_argument("--output", default=None)
    parser.add_argument("--fetch-pdf-text", action="store_true", help="Download arXiv PDFs and extract the first pages.")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["commons", "arxiv", "archive", "nasa"],
        choices=["commons", "arxiv", "archive", "nasa", *BULK_IA_SOURCES],
    )
    args = parser.parse_args()

    settings = get_settings()
    output = settings.root / args.output if args.output else settings.docs_path
    console = Console()
    records = []
    with OpenSourceClient(settings.user_agent, timeout_s=settings.opensearch_timeout_s) as client:
        if args.bulk_internet_archive:
            target_docs = args.target_docs or args.limit_per_source * 4
            records.extend(_build_bulk_internet_archive(client, args, target_docs, console))
        else:
            fetchers = {
                "commons": lambda: client.wikimedia_images(args.query, args.limit_per_source),
                "arxiv": lambda: client.arxiv_pdfs(args.query, args.limit_per_source, fetch_pdf_text=args.fetch_pdf_text),
                "archive": lambda: client.internet_archive_videos(args.query, args.limit_per_source),
                "nasa": lambda: client.nasa_exoplanet_rows(args.limit_per_source),
            }
            for name in args.sources:
                try:
                    batch = fetchers[name]()
                    records.extend(batch)
                    console.print(f"{name}: {len(batch):,} records")
                except Exception as exc:
                    console.print(f"[yellow]{name}: skipped ({type(exc).__name__}: {exc})[/yellow]")

    records = dedupe_records(records)
    write_jsonl(output, [record.model_dump(mode="json") for record in records])
    console.print(f"Saved {len(records):,} records -> {output}")
    return 0


def _build_bulk_internet_archive(
    client: OpenSourceClient,
    args: argparse.Namespace,
    target_docs: int,
    console: Console,
):
    active_sources = [name for name in args.sources if name in BULK_IA_SOURCES] or list(BULK_IA_SOURCES)
    table_records = []
    if "nasa" in args.sources and target_docs > 1000:
        try:
            table_limit = min(max(target_docs // 20, args.limit_per_source), 5000)
            table_records = client.nasa_exoplanet_rows(table_limit)
            console.print(f"nasa: {len(table_records):,} SQL-style table records")
        except Exception as exc:
            console.print(f"[yellow]nasa: skipped ({type(exc).__name__}: {exc})[/yellow]")
    records = list(table_records)
    ia_target = max(0, target_docs - len(table_records))
    quota = max(1, ia_target // len(active_sources))
    remainder = max(0, ia_target - quota * len(active_sources))
    for index, name in enumerate(active_sources):
        mediatype, modality = BULK_IA_SOURCES[name]
        limit = quota + (1 if index < remainder else 0)
        try:
            batch = client.internet_archive_media_sharded(
                mediatype,
                modality,
                query=args.ia_query,
                limit=limit,
                page_size=args.ia_page_size,
            )
            records.extend(batch)
            console.print(f"{name}: {len(batch):,} records")
        except Exception as exc:
            console.print(f"[yellow]{name}: skipped ({type(exc).__name__}: {exc})[/yellow]")
    shortfall = max(0, target_docs - len(dedupe_records(records)))
    if shortfall:
        console.print(f"[yellow]bulk shortfall: fetching {shortfall:,} additional text records[/yellow]")
        records.extend(
            client.internet_archive_media_sharded(
                "texts",
                "pdf",
                query=args.ia_query,
                limit=shortfall,
                page_size=args.ia_page_size,
            )
        )
    return dedupe_records(records)[:target_docs]


if __name__ == "__main__":
    raise SystemExit(main())
