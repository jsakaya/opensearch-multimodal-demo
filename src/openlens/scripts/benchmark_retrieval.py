from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from openlens.config import get_settings
from openlens.data import utc_now
from openlens.indexer import check_status, make_client
from openlens.retrieval import SearchMode, make_retriever
from openlens.text import clean_text


MODES: tuple[SearchMode, ...] = ("keyword", "vector", "hybrid", "lir")
MODALITIES = ("audio", "image", "pdf", "table", "video")


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    query: str
    expected_doc_id: str = ""
    expected_modality: str = ""
    mode: SearchMode = "hybrid"
    kind: str = "exact"


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark OpenLens retrieval latency and quality on OpenSearch.")
    parser.add_argument("--samples-per-modality", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=80)
    parser.add_argument("--output", default="")
    parser.add_argument("--include-sql", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    settings = get_settings()
    status = check_status(settings)
    if not status.available or status.doc_count <= 0:
        raise SystemExit(f"OpenSearch benchmark requires a populated index: {status.detail}")
    retriever = make_retriever(settings, prefer_opensearch=True)
    if retriever.__class__.__name__ != "OpenSearchRetriever":
        raise SystemExit("Benchmark must run against OpenSearchRetriever.")

    cases = exact_cases(settings.opensearch_index, args.samples_per_modality)
    cases.extend(scenario_cases())
    if args.include_sql:
        cases.extend(sql_cases())

    for case in cases[: max(0, args.warmups)]:
        retriever.search(case.query, mode=case.mode, top_k=args.top_k, candidate_k=args.candidate_k)

    runs: list[dict[str, Any]] = []
    for case in cases:
        for repeat in range(args.repeats):
            started = time.perf_counter()
            response = retriever.search(case.query, mode=case.mode, top_k=args.top_k, candidate_k=args.candidate_k)
            wall_ms = (time.perf_counter() - started) * 1000
            hit_ids = [hit.doc_id for hit in response.hits]
            hit_modalities = [str(hit.doc.get("modality") or "") for hit in response.hits]
            runs.append(
                {
                    "case_id": case.case_id,
                    "kind": case.kind,
                    "repeat": repeat + 1,
                    "query": case.query,
                    "mode": case.mode,
                    "latency_ms": response.latency_ms,
                    "wall_ms": round(wall_ms, 2),
                    "total": len(response.hits),
                    "top_doc_id": hit_ids[0] if hit_ids else "",
                    "top_modality": hit_modalities[0] if hit_modalities else "",
                    "expected_doc_id": case.expected_doc_id,
                    "expected_modality": case.expected_modality,
                    "exact_at_k": bool(case.expected_doc_id and case.expected_doc_id in hit_ids),
                    "modality_at_1": bool(case.expected_modality and hit_modalities[:1] == [case.expected_modality]),
                    "modality_at_3": bool(case.expected_modality and case.expected_modality in hit_modalities[:3]),
                }
            )

    payload = {
        "generated_at": utc_now(),
        "settings": {
            "index": settings.opensearch_index,
            "vector_dim": settings.vector_dim,
            "embedding_backend": settings.embedding_backend,
            "qwen_model": settings.qwen_model,
            "qwen_batch_size": settings.qwen_batch_size,
            "qwen_max_frames": settings.qwen_max_frames,
        },
        "opensearch": status.__dict__,
        "summary": summarize(runs),
        "runs": runs,
    }
    console = Console()
    print_summary(console, payload["summary"])
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        output.with_suffix(".md").write_text(markdown_report(payload), encoding="utf-8")
        console.print(f"wrote {output}")
        console.print(f"wrote {output.with_suffix('.md')}")
    else:
        console.print_json(json.dumps(payload["summary"]))
    return 0


def exact_cases(index_name: str, samples_per_modality: int) -> list[BenchmarkCase]:
    settings = get_settings()
    client = make_client(settings)
    cases: list[BenchmarkCase] = []
    for modality in MODALITIES:
        body = {
            "size": samples_per_modality,
            "query": {"term": {"modality": modality}},
            "sort": [{"doc_id": "asc"}],
            "_source": {"excludes": ["vector", "patch_vectors"]},
        }
        response = client.search(index=index_name, body=body)
        for idx, hit in enumerate(response.get("hits", {}).get("hits", []), start=1):
            doc = hit.get("_source", {})
            query = query_from_doc(doc)
            if not query:
                continue
            for mode in MODES:
                cases.append(
                    BenchmarkCase(
                        case_id=f"exact-{modality}-{idx}-{mode}",
                        query=query,
                        expected_doc_id=str(doc.get("doc_id") or hit.get("_id") or ""),
                        expected_modality=modality,
                        mode=mode,
                    )
                )
    return cases


def query_from_doc(doc: dict[str, Any]) -> str:
    title = clean_text(doc.get("title"), max_chars=120)
    tags = [str(tag) for tag in (doc.get("tags") or []) if str(tag).strip()]
    summary = clean_text(doc.get("summary") or doc.get("body"), max_chars=120)
    modality = str(doc.get("modality") or "")
    if modality == "audio":
        return clean_text(f"audio recording {title} {' '.join(tags[:5])} {summary}", max_chars=260)
    if modality == "video":
        return clean_text(f"video clip {title} {' '.join(tags[:5])} {summary}", max_chars=260)
    if modality == "table":
        return clean_text(f"table row {title} {summary}", max_chars=260)
    return clean_text(f"{title} {' '.join(tags[:5])} {summary}", max_chars=260)


def scenario_cases() -> list[BenchmarkCase]:
    scenarios = [
        ("scenario-audio", "NASA mission control audio schedule inventory", "audio"),
        ("scenario-video", "Artemis moon landing rocket launch spacecraft video", "video"),
        ("scenario-image", "Mars rover moon earth satellite NASA image", "image"),
        ("scenario-pdf", "NASA technical report about earth observation exoplanet climate", "pdf"),
        ("scenario-table", "transit method exoplanet orbital period table row", "table"),
    ]
    cases: list[BenchmarkCase] = []
    for case_id, query, modality in scenarios:
        for mode in MODES:
            cases.append(BenchmarkCase(case_id=f"{case_id}-{mode}", query=query, expected_modality=modality, mode=mode, kind="scenario"))
    return cases


def sql_cases() -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            case_id="sql-modality-counts",
            query="SELECT modality, COUNT(*) AS n FROM openlens GROUP BY modality",
            expected_modality="table",
            mode="sql",
            kind="sql",
        ),
        BenchmarkCase(
            case_id="sql-exoplanet-sample",
            query="SELECT title, source_id FROM openlens WHERE modality = 'table' LIMIT 10",
            expected_modality="table",
            mode="sql",
            kind="sql",
        ),
    ]


def summarize(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_mode[str(run["mode"])].append(run)
        by_kind[str(run["kind"])].append(run)
    return {
        "all": aggregate(runs),
        "by_mode": {mode: aggregate(rows) for mode, rows in sorted(by_mode.items())},
        "by_kind": {kind: aggregate(rows) for kind, rows in sorted(by_kind.items())},
        "by_expected_modality": summarize_expected_modality(runs),
        "by_mode_and_expected_modality": summarize_mode_and_expected_modality(runs),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in rows]
    exact_rows = [row for row in rows if row.get("expected_doc_id")]
    modality_rows = [row for row in rows if row.get("expected_modality")]
    return {
        "count": len(rows),
        "latency_avg_ms": round(statistics.fmean(latencies), 2) if latencies else 0,
        "latency_p50_ms": round(percentile(latencies, 50), 2),
        "latency_p95_ms": round(percentile(latencies, 95), 2),
        "exact_at_k": round(rate(row.get("exact_at_k") for row in exact_rows), 4),
        "modality_at_1": round(rate(row.get("modality_at_1") for row in modality_rows), 4),
        "modality_at_3": round(rate(row.get("modality_at_3") for row in modality_rows), 4),
    }


def summarize_expected_modality(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_modality: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        modality = str(run.get("expected_modality") or "")
        if modality:
            by_modality[modality].append(run)
    return {modality: aggregate(rows) for modality, rows in sorted(by_modality.items())}


def summarize_mode_and_expected_modality(runs: list[dict[str, Any]]) -> dict[str, Any]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        modality = str(run.get("expected_modality") or "")
        mode = str(run.get("mode") or "")
        if modality and mode:
            by_pair[f"{modality}:{mode}"].append(run)
    return {key: aggregate(rows) for key, rows in sorted(by_pair.items())}


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil((pct / 100) * len(ordered)) - 1))
    return ordered[index]


def rate(values: Any) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(1 for item in items if item) / len(items)


def print_summary(console: Console, summary: dict[str, Any]) -> None:
    table = Table(title="OpenLens OpenSearch retrieval benchmark")
    table.add_column("Slice")
    table.add_column("n", justify="right")
    table.add_column("p50 ms", justify="right")
    table.add_column("p95 ms", justify="right")
    table.add_column("exact@k", justify="right")
    table.add_column("mod@1", justify="right")
    table.add_column("mod@3", justify="right")
    table.add_row("all", *row_values(summary["all"]))
    for mode, row in summary["by_mode"].items():
        table.add_row(mode, *row_values(row))
    console.print(table)


def row_values(row: dict[str, Any]) -> list[str]:
    return [
        str(row["count"]),
        f"{row['latency_p50_ms']:.2f}",
        f"{row['latency_p95_ms']:.2f}",
        f"{row['exact_at_k']:.2%}",
        f"{row['modality_at_1']:.2%}",
        f"{row['modality_at_3']:.2%}",
    ]


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# OpenLens Retrieval Benchmark",
        "",
        f"- Generated: `{payload['generated_at']}`",
        f"- Index: `{payload['settings']['index']}`",
        f"- OpenSearch: `{payload['opensearch']['detail']}`",
        f"- Docs: `{payload['opensearch']['doc_count']}`",
        f"- Embeddings: `{payload['settings']['embedding_backend']}` `{payload['settings']['vector_dim']}d`",
        "",
        "| Slice | n | p50 ms | p95 ms | exact@k | modality@1 | modality@3 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    summary = payload["summary"]
    lines.append("| all | " + " | ".join(row_values(summary["all"])) + " |")
    for mode, row in summary["by_mode"].items():
        lines.append(f"| {mode} | " + " | ".join(row_values(row)) + " |")
    lines.extend(["", "## By Expected Modality", "", "| Modality | n | p50 ms | p95 ms | exact@k | modality@1 | modality@3 |", "|---|---:|---:|---:|---:|---:|---:|"])
    for modality, row in summary["by_expected_modality"].items():
        lines.append(f"| {modality} | " + " | ".join(row_values(row)) + " |")
    lines.extend(["", "## By Mode And Expected Modality", "", "| Modality | Mode | n | p50 ms | p95 ms | exact@k | modality@1 | modality@3 |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for key, row in summary["by_mode_and_expected_modality"].items():
        modality, mode = key.split(":", 1)
        lines.append(f"| {modality} | {mode} | " + " | ".join(row_values(row)) + " |")
    lines.extend(["", "## Notes", "", "- `exact@k` uses title/metadata queries generated from indexed records.", "- `modality@1` and `modality@3` measure whether scenario queries retrieve the expected modality.", "- SQL rows use the OpenSearch SQL plugin path."])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
