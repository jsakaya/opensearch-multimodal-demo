from __future__ import annotations

import argparse

from rich.console import Console

from openlens.config import get_settings
from openlens.data import write_jsonl
from openlens.sources import DEFAULT_QUERY, OpenSourceClient, dedupe_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a small open multimodal corpus from public sources.")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--limit-per-source", type=int, default=8)
    parser.add_argument("--output", default=None)
    parser.add_argument("--fetch-pdf-text", action="store_true", help="Download arXiv PDFs and extract the first pages.")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["commons", "arxiv", "archive", "nasa"],
        choices=["commons", "arxiv", "archive", "nasa"],
    )
    args = parser.parse_args()

    settings = get_settings()
    output = settings.root / args.output if args.output else settings.docs_path
    console = Console()
    records = []
    with OpenSourceClient(settings.user_agent, timeout_s=settings.opensearch_timeout_s) as client:
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


if __name__ == "__main__":
    raise SystemExit(main())
