#!/usr/bin/env python3
"""Compare paired per-query evaluation files, optionally by query category.

Example:
    uv run python eval/analyze_p3.py \
      eval/results/perquery_full_no_rerank.jsonl \
      eval/results/perquery_full_rerank.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

METRICS = ("Recall@50", "nDCG@10", "MRR", "P@5", "ILD@5")


def load(path: Path) -> dict[str, dict]:
    return {
        row["query_id"]: row
        for row in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("reranked", type=Path)
    args = parser.parse_args()
    baseline, reranked = load(args.baseline), load(args.reranked)
    common = sorted(set(baseline) & set(reranked))
    groups: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for query_id in common:
        groups[reranked[query_id].get("category") or "uncategorized"].append(
            (baseline[query_id], reranked[query_id])
        )

    print(f"paired queries: {len(common)}")
    print("| Category | N | " + " | ".join(f"Δ {m}" for m in METRICS) + " |")
    print("|---|---:|" + "---:|" * len(METRICS))
    for category, rows in sorted(groups.items()):
        deltas = [sum(b[m] - a[m] for a, b in rows) / len(rows) for m in METRICS]
        print("| " + category + " | " + str(len(rows)) + " | " + " | ".join(
            f"{value:+.3f}" for value in deltas
        ) + " |")


if __name__ == "__main__":
    main()
