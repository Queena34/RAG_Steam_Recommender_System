#!/usr/bin/env python3
"""Sweep QUALITY_WEIGHT and watch two things move at once.

The judge run showed the revised system recommending titles with a median of
468 positive reviews against the original's 8,935, and scoring 8 points lower
for it. The retrieval metrics could not see this: they ask whether tags match
or whether a specific game came back, not whether a game is worth recommending.

So this sweep reports both together --- how well-reviewed the shortlist is, and
whether topical relevance holds up --- because moving only one of them is what
produced the regression in the first place.

Retrieval is unchanged by the weight, so candidates are fetched once per query
and re-scored for each setting. That makes the sweep cheap: one pass of
embedding and BM25, then arithmetic.

    uv run python eval/sweep_quality_weight.py
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics as st
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics as M  # noqa: E402
import recommender as R  # noqa: E402
import run_eval as RE  # noqa: E402

HERE = Path(__file__).resolve().parent


def rescore(records, quality_weight: float) -> list:
    """Re-apply the relevance/quality blend at a given weight."""
    if not records:
        return []
    max_rrf = max(r.raw.get("_rrf", 0.0) for r in records) or 1.0
    for rec in records:
        relevance = rec.raw.get("_rrf", 0.0) / max_rrf
        quality = R.GameSearchEngine._quality_score(rec)
        rec.raw["_score"] = (1 - quality_weight) * relevance + quality_weight * quality
    return sorted(records, key=lambda r: r.raw["_score"], reverse=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=float, nargs="+",
                    default=[0.0, 0.25, 0.4, 0.5, 0.6, 0.75])
    ap.add_argument("--queries", type=Path, default=HERE / "queries_topical.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE / "results" / "quality_weight_sweep.md")
    args = ap.parse_args()

    queries = [json.loads(l) for l in args.queries.read_text().splitlines() if l.strip()]
    if args.limit:
        queries = queries[: args.limit]

    engine = R.GameSearchEngine(R.DB_PATH)
    conn = sqlite3.connect(R.DB_PATH)

    # Retrieve once per query; the weight only affects ordering afterwards.
    print(f"Retrieving candidates for {len(queries)} queries...")
    per_query = []
    for i, q in enumerate(queries, 1):
        app_ids = RE.retrieve_rrf(engine, q["query"])
        records = RE._hydrate(engine, app_ids) if app_ids else []
        per_query.append((q, records))
        if i % 10 == 0:
            print(f"  {i}/{len(queries)}")

    rows = []
    for w in args.weights:
        stats = {"nDCG@10": [], "P@5": [], "MRR": []}
        medians = []
        for q, records in per_query:
            ordered = rescore(records, w)
            ids = [r.app_id for r in ordered]
            relevant, grades = RE.judgements(q)
            stats["nDCG@10"].append(M.ndcg_at_k(ids, grades, 10))
            stats["P@5"].append(M.precision_at_k(ids, relevant, 5))
            stats["MRR"].append(M.reciprocal_rank(ids, relevant))

            top5 = ids[:5]
            if top5:
                ph = ",".join("?" * len(top5))
                pos = [
                    r[0] for r in conn.execute(
                        f"SELECT positive FROM games WHERE appid IN ({ph})",
                        [int(a) for a in top5],
                    ) if r[0] is not None
                ]
                if pos:
                    medians.append(st.median(pos))

        summary = M.summarise(stats)
        median_reviews = st.median(medians) if medians else 0
        rows.append((w, summary, median_reviews))
        print(f"  w={w:.2f}  nDCG@10={summary['nDCG@10'][0]:.3f}  "
              f"P@5={summary['P@5'][0]:.3f}  median reviews={int(median_reviews):,}")

    lines = [
        "# QUALITY_WEIGHT sweep",
        "",
        f"- {len(queries)} topical queries",
        "- `median reviews` is the median positive-review count of the top five",
        "- Retrieval is identical across rows; only the blend weight changes",
        "",
        "| Weight | nDCG@10 | P@5 | MRR | Median reviews in top 5 |",
        "|---|---|---|---|---|",
    ]
    for w, s, med in rows:
        lines.append(
            f"| {w:.2f} | {s['nDCG@10'][0]:.3f} ± {s['nDCG@10'][1]:.3f} "
            f"| {s['P@5'][0]:.3f} ± {s['P@5'][1]:.3f} "
            f"| {s['MRR'][0]:.3f} ± {s['MRR'][1]:.3f} "
            f"| {int(med):,} |"
        )
    lines += [
        "",
        "For reference, the judge run recorded a median of 8,935 positive reviews "
        "for the original system's picks, 468 for the revised system's, and "
        "37,721 for ChatGPT's.",
    ]
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
