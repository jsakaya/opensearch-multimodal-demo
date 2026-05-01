#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from openlens.data import read_jsonl


ROOT = Path(__file__).resolve().parents[3]


def runpod_key() -> str:
    if os.getenv("RUNPOD_API_KEY"):
        return os.environ["RUNPOD_API_KEY"]
    try:
        value = subprocess.check_output(
            ["security", "find-generic-password", "-s", "runpod-api-key", "-w"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        value = ""
    if not value:
        raise SystemExit("missing RUNPOD_API_KEY; export it or store it in Keychain as runpod-api-key")
    return value


def endpoint_id(explicit: str | None) -> str:
    if explicit:
        return explicit
    if os.getenv("RUNPOD_ENDPOINT_ID"):
        return os.environ["RUNPOD_ENDPOINT_ID"]
    path = ROOT / ".runpod" / "serverless-endpoint-id"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    raise SystemExit("missing endpoint id; run make serverless-deploy or set RUNPOD_ENDPOINT_ID")


def wait_for_job(client: httpx.Client, api_key: str, endpoint: str, initial: dict[str, Any]) -> dict[str, Any]:
    status = str(initial.get("status") or "")
    if status not in {"IN_QUEUE", "IN_PROGRESS"}:
        return initial
    job_id = str(initial.get("id") or "")
    if not job_id:
        return initial
    timeout_s = int(os.getenv("RUNPOD_SERVERLESS_WAIT_S", "1800"))
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(5)
        response = client.get(
            f"https://api.runpod.ai/v2/{endpoint}/status/{job_id}",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        response.raise_for_status()
        current = response.json()
        if current.get("status") not in {"IN_QUEUE", "IN_PROGRESS"}:
            return current
        print(f"waiting for {job_id}: {current.get('status')}", file=sys.stderr)
    raise SystemExit(f"timed out waiting for RunPod job {job_id}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Submit an explicit one-record RunPod Serverless encoder smoke job.")
    parser.add_argument("--endpoint-id", default=None)
    parser.add_argument("--docs", type=Path, default=ROOT / "data/processed/open_corpus.jsonl")
    parser.add_argument("--records", type=int, default=1)
    parser.add_argument("--backend", default=os.getenv("OPENLENS_EMBEDDING_BACKEND", "colpali"))
    parser.add_argument("--model", default=os.getenv("OPENLENS_COLPALI_MODEL", "colpali-v1.3"))
    parser.add_argument("--action", default="encode", choices=["encode", "status"])
    args = parser.parse_args()

    api_key = runpod_key()
    endpoint = endpoint_id(args.endpoint_id)

    if args.action == "status":
        payload: dict[str, Any] = {"input": {"action": "status"}}
    else:
        rows = read_jsonl(args.docs)[: args.records]
        if not rows:
            raise SystemExit(f"no rows found in {args.docs}")
        payload = {
            "input": {
                "backend": args.backend,
                "colpali_model": args.model,
                "records": rows,
                "return_records": True,
                "inline_max_records": max(args.records, 1),
            }
        }

    with httpx.Client(timeout=httpx.Timeout(None, connect=10.0)) as client:
        response = client.post(
            f"https://api.runpod.ai/v2/{endpoint}/runsync",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        result = wait_for_job(client, api_key, endpoint, response.json())

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
