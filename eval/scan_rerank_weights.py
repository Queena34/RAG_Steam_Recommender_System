#!/usr/bin/env python3
"""Scan relevance/quality blends after one Cross-encoder pass per query."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, default=Path(__file__).parent / "queries_knownitem.jsonl")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--weights", default="0.2,0.4,0.6,0.8")
    args = ap.parse_args()
    from recommender import DB_PATH, GameSearchEngine

    weights = [float(x) for x in args.weights.split(",")]
    queries = [json.loads(x) for x in args.queries.read_text().splitlines() if x][:args.limit]
    engine = GameSearchEngine(DB_PATH)
    scores = {w: {m: [] for m in ("nDCG@10", "MRR", "P@5")} for w in weights}
    for query in queries:
        candidates = engine.retrieve_candidates(query["query"])
        cross = engine._rerank_cross_encoder(query["query"], candidates)
        if not cross:
            continue
        for rec, relevance in cross:
            rec.raw["_ce_score"] = relevance
        by_id = {rec.app_id: rec for rec, _ in cross}
        for weight in weights:
            for rec in by_id.values():
                rec.raw["_score"] = weight * rec.raw.get("_ce_score", 0.0) + (1 - weight) * GameSearchEngine._quality_score(rec)
            ranked = sorted(by_id.values(), key=lambda rec: rec.raw["_score"], reverse=True)
            ranked = engine._apply_diversity([(r, r.raw["_score"]) for r in ranked], 10)
            ids = [r.app_id for r, _ in ranked]
            relevant = {str(query["gold_appid"])}
            grades = {str(query["gold_appid"]): 2.0}
            scores[weight]["nDCG@10"].append(M.ndcg_at_k(ids, grades, 10))
            scores[weight]["MRR"].append(M.reciprocal_rank(ids, relevant))
            scores[weight]["P@5"].append(M.precision_at_k(ids, relevant, 5))
    print(f"queries_scored={len(next(iter(scores.values()))['MRR'])}")
    print("| CE weight | nDCG@10 | MRR | P@5 |")
    print("|---:|---:|---:|---:|")
    for weight in weights:
        row = scores[weight]
        print("| %.2f | %.3f | %.3f | %.3f |" % tuple(
            [weight] + [sum(row[m]) / len(row[m]) if row[m] else 0.0 for m in ("nDCG@10", "MRR", "P@5")]
        ))


if __name__ == "__main__":
    main()
