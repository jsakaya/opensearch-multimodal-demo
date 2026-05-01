from __future__ import annotations

import argparse

from rich.console import Console

from openlens.config import get_settings
from openlens.data import read_records
from openlens.indexer import embed_and_optionally_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed and index OpenLens records into OpenSearch.")
    parser.add_argument("--input", default=None)
    parser.add_argument("--skip-opensearch", action="store_true")
    parser.add_argument("--no-recreate", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    input_path = settings.root / args.input if args.input else settings.docs_path
    console = Console()
    records = read_records(input_path)
    if not records:
        raise SystemExit(f"No records found at {input_path}. Run `uv run openlens-build` first.")
    indexed, status = embed_and_optionally_index(
        settings=settings,
        records=records,
        recreate=not args.no_recreate,
        skip_opensearch=args.skip_opensearch,
    )
    console.print(f"Embedded {len(indexed):,} records -> {settings.embedded_docs_path}")
    console.print(status.detail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
