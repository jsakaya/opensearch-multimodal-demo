from __future__ import annotations

import argparse

from rich.console import Console

from openlens.config import get_settings
from openlens.data import write_jsonl
from openlens.sources import DEFAULT_QUERY, OpenSourceClient, dedupe_records


NASA_DEMO_SOURCES = ("nasa-images", "nasa-videos", "nasa-audio", "ntrs", "nasa-exoplanets")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the OpenLens NASA multimodal corpus.")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--limit-per-source", type=int, default=8)
    parser.add_argument("--target-docs", type=int, default=0, help="Target record count for the NASA demo build.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--fetch-pdf-text", action="store_true", help="Download available NASA STI full-text snippets.")
    parser.add_argument(
        "--customer-demo-space",
        action="store_true",
        help="Build the full NASA/space customer demo corpus. This is the default when --target-docs is set.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(NASA_DEMO_SOURCES),
        choices=[
            "nasa",
            "nasa-images",
            "nasa-videos",
            "nasa-audio",
            "nasa-exoplanets",
            "ntrs",
        ],
    )
    args = parser.parse_args()

    settings = get_settings()
    output = settings.root / args.output if args.output else settings.docs_path
    console = Console()
    records = []
    with OpenSourceClient(settings.user_agent, timeout_s=settings.opensearch_timeout_s) as client:
        if args.customer_demo_space or args.target_docs:
            records.extend(_build_customer_demo_space(client, args, console))
        else:
            demo_queries = _space_queries(args.query)
            fetchers = {
                "nasa": lambda: client.nasa_exoplanet_rows(args.limit_per_source),
                "nasa-exoplanets": lambda: client.nasa_exoplanet_rows(args.limit_per_source),
                "nasa-images": lambda: _fetch_nasa_media_mix(client, demo_queries, "image", args.limit_per_source),
                "nasa-videos": lambda: _fetch_nasa_media_mix(client, demo_queries, "video", args.limit_per_source),
                "nasa-audio": lambda: _fetch_nasa_media_mix(client, demo_queries, "audio", args.limit_per_source),
                "ntrs": lambda: _fetch_ntrs_mix(
                    client,
                    demo_queries,
                    args.limit_per_source,
                    fetch_pdf_text=args.fetch_pdf_text,
                ),
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


def _build_customer_demo_space(client: OpenSourceClient, args: argparse.Namespace, console: Console):
    target_docs = args.target_docs or 10000
    quotas = {
        "nasa-images": int(target_docs * 0.35),
        "nasa-videos": int(target_docs * 0.18),
        "nasa-audio": int(target_docs * 0.05),
        "ntrs": int(target_docs * 0.24),
        "nasa-exoplanets": target_docs,
    }
    quotas["nasa-exoplanets"] = max(1, target_docs - sum(value for key, value in quotas.items() if key != "nasa-exoplanets"))
    query = args.query or "artemis moon mars earth exoplanet"
    media_queries = _space_queries(query)
    records = []
    fetchers = {
        "nasa-images": lambda limit: _fetch_nasa_media_mix(client, media_queries, "image", limit),
        "nasa-videos": lambda limit: _fetch_nasa_media_mix(client, media_queries, "video", limit),
        "nasa-audio": lambda limit: _fetch_nasa_media_mix(client, media_queries, "audio", limit),
        "ntrs": lambda limit: _fetch_ntrs_mix(client, media_queries, limit, fetch_pdf_text=args.fetch_pdf_text),
        "nasa-exoplanets": lambda limit: client.nasa_exoplanet_rows(limit),
    }
    for name, limit in quotas.items():
        try:
            batch = fetchers[name](limit)
            records.extend(batch)
            console.print(f"{name}: {len(batch):,} records")
        except Exception as exc:
            console.print(f"[yellow]{name}: skipped ({type(exc).__name__}: {exc})[/yellow]")
    shortfall = max(0, target_docs - len(dedupe_records(records)))
    if shortfall:
        console.print(f"[yellow]space demo shortfall: fetching {shortfall:,} extra NASA images[/yellow]")
        records.extend(_fetch_nasa_media_mix(client, media_queries, "image", shortfall))
    final_records = dedupe_records(records)
    shortfall = max(0, target_docs - len(final_records))
    if shortfall:
        console.print(f"[yellow]space demo shortfall: fetching {shortfall:,} extra exoplanet rows[/yellow]")
        seen = {record.doc_id for record in final_records}
        extra_tables = client.nasa_exoplanet_rows(quotas["nasa-exoplanets"] + shortfall + 100)
        records.extend(record for record in extra_tables if record.doc_id not in seen)
    return dedupe_records(records)[:target_docs]


def _space_queries(query: str) -> list[str]:
    base = [part.strip() for part in query.replace(",", " ").split() if part.strip()]
    defaults = [
        "artemis",
        "apollo",
        "mars",
        "earth",
        "moon",
        "hubble",
        "webb",
        "exoplanet",
        "spacewalk",
        "audio",
        "podcast",
        "hwhap",
        "houston we have a podcast",
        "this week at nasa",
        "launch",
        "landing",
        "crew",
        "astronaut",
        "flight director",
        "kennedy",
        "johnson space center",
        "mission control",
        "international space station",
        "earth observation",
        "james webb space telescope",
        "perseverance rover",
        "climate",
        "galaxy",
        "nebula",
        "satellite",
        "space shuttle",
        "kennedy space center",
        "hubble telescope",
        "solar system",
        "jupiter",
        "saturn",
        "orion spacecraft",
        "sls rocket",
        "iss",
        "astronaut training",
        "lunar",
        "aeronautics",
        "x-59",
        "telescope",
        "star cluster",
        "black hole",
        "supernova",
        "rocket launch",
        "apollo 11",
        "earth from space",
        "spacecraft",
    ]
    out: list[str] = []
    for item in [*base, *defaults]:
        lowered = item.lower()
        if lowered not in out:
            out.append(lowered)
    return out


def _fetch_nasa_media_mix(
    client: OpenSourceClient,
    queries: list[str],
    media_type: str,
    limit: int,
) -> list:
    records = []
    seen: set[str] = set()
    per_query = max(25, min(100, limit))
    for query in queries:
        if len(records) >= limit:
            break
        for record in client.nasa_media(query, media_type, per_query):
            if record.doc_id not in seen:
                records.append(record)
                seen.add(record.doc_id)
            if len(records) >= limit:
                break
    return records[:limit]


def _fetch_ntrs_mix(
    client: OpenSourceClient,
    queries: list[str],
    limit: int,
    fetch_pdf_text: bool = False,
) -> list:
    records = []
    seen: set[str] = set()
    per_query = max(25, min(100, limit // max(1, len(queries)) + 10))
    for query in queries:
        if len(records) >= limit:
            break
        for record in client.ntrs_pdfs(query, per_query, fetch_pdf_text=fetch_pdf_text):
            if record.doc_id not in seen:
                records.append(record)
                seen.add(record.doc_id)
            if len(records) >= limit:
                break
    return records[:limit]


if __name__ == "__main__":
    raise SystemExit(main())
