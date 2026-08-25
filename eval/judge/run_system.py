#!/usr/bin/env python3
"""Capture the revised system's recommendations for the judge protocol.

Data collection only -- nothing here scores anything. The scoring pass has to
be done by a judge with no stake in the system, which rules out whoever built
it; see eval/judge/README.md.

Each round records the full API response so the telemetry added in PRD-004
(generation mode, rejected identifiers, evidence support) can be reported
alongside the judge scores.

    uv run python eval/judge/run_system.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

HERE = Path(__file__).resolve().parent

from recommender import GameSearchEngine, DB_PATH  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, default=HERE / "prompts.jsonl")
    ap.add_argument("--out", type=Path, default=HERE / "system_outputs.jsonl")
    ap.add_argument("--runs", type=int, default=1,
                    help="repeat each prompt; generation is not deterministic")
    args = ap.parse_args()

    prompts = [json.loads(l) for l in args.prompts.read_text().splitlines() if l.strip()]
    engine = GameSearchEngine(DB_PATH)
    print(f"\nRounds: {len(prompts)}   runs each: {args.runs}\n")

    records = []
    for p in prompts:
        for run in range(1, args.runs + 1):
            started = time.perf_counter()
            try:
                result = engine.search(p["query"])
                err = None
            except Exception as exc:
                result, err = None, str(exc)
            elapsed = time.perf_counter() - started

            rec = {
                "round": p["round"],
                "category": p["category"],
                "query": p["query"],
                "run": run,
                "latency": elapsed,
                "error": err,
            }
            if result:
                rec["titles"] = [m["name"] for m in result["matches"]]
                rec["app_ids"] = [m["app_id"] for m in result["matches"]]
                rec["answer"] = result["answer"]
                rec["meta"] = result["meta"]
            records.append(rec)

            mode = (result or {}).get("meta", {}).get("generation_mode", "-")
            titles = ", ".join((rec.get("titles") or [])[:3])
            print(f"  R{p['round']} run {run}  {elapsed:6.1f}s  {mode:20s} {titles}")

    args.out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
    )
    print(f"\nWritten to {args.out}")

    modes = {}
    for r in records:
        m = (r.get("meta") or {}).get("generation_mode", "error")
        modes[m] = modes.get(m, 0) + 1
    rejected = sum((r.get("meta") or {}).get("rejected_app_ids", 0) for r in records)
    supports = [
        (r.get("meta") or {}).get("evidence_supported")
        for r in records
        if (r.get("meta") or {}).get("evidence_supported") is not None
    ]
    print("\nTelemetry")
    for m, n in sorted(modes.items()):
        print(f"  {m:22s} {n}")
    print(f"  identifiers rejected   {rejected}")
    if supports:
        print(f"  evidence supported     {sum(supports)/len(supports):.2f} mean")


if __name__ == "__main__":
    main()
