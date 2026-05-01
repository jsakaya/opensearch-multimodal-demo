from __future__ import annotations

import argparse

from rich.console import Console

from openlens.config import get_settings
from openlens.retrieval import make_retriever


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a retrieval smoke test against OpenLens.")
    parser.add_argument("--query", default="satellite images of earth climate change")
    parser.add_argument("--mode", choices=["hybrid", "keyword", "vector", "lir"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    retriever = make_retriever(settings, prefer_opensearch=not args.local)
    response = retriever.search(args.query, mode=args.mode, top_k=args.top_k)
    console = Console()
    console.rule(f"{response.retriever} {response.mode} search")
    for hit in response.hits:
        console.print(f"{hit.rank}. [{hit.doc.get('modality')}] {hit.doc.get('title')} score={hit.score:.4f}")
        console.print(f"   {hit.excerpt}")
    console.print(f"latency_ms={response.latency_ms:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
