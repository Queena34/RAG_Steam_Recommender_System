#!/usr/bin/env python3
"""Ablation harness for the retrieval and ranking stages (PRD-002, R5 and R7).

Each configuration differs from the one above it by exactly one component, so
the difference between two rows is that component's contribution:

    bm25        lexical baseline
    vector      semantic baseline
    rrf         + rank fusion                      -> C1
    full        + review-aware quality, diversity  -> C5 / C6

Configurations are assembled here from the engine's retrieval primitives
rather than by adding a mode switch to retrieve_candidates(). Evaluation
concerns stay out of the production path, and adding a configuration never
means editing production logic.

Point RAGLOOKER_INDEX_DIR at an index directory to compare corpora:

    uv run python eval/run_eval.py --configs bm25,vector,rrf,full
    RAGLOOKER_INDEX_DIR=vector_index_baseline_18k \\
        uv run python eval/run_eval.py --configs vector --label "vector (18k)"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics as M  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Depth each retriever contributes, matching RETRIEVAL_POOL in production.
POOL = 100
# Depth the fused candidate list is cut to, matching RETRIEVAL_COUNT.
CUT = 50


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

def retrieve_bm25(engine, query: str) -> list[str]:
    return engine._retrieve_bm25_ids(query, POOL)[:CUT]


def retrieve_vector(engine, query: str) -> list[str]:
    return engine._retrieve_vector_ids(query, POOL)[:CUT]


def retrieve_rrf(engine, query: str) -> list[str]:
    lists = [
        lst for lst in (
            engine._retrieve_vector_ids(query, POOL),
            engine._retrieve_bm25_ids(query, POOL),
        ) if lst
    ]
    if not lists:
        return []
    return [app_id for app_id, _ in engine._rrf_fuse(lists)][:CUT]


def retrieve_full(engine, query: str) -> list[str]:
    """Fusion, then the production scoring and diversity pass.

    Reuses retrieve_rrf for the candidate set, so this configuration cannot
    retrieve anything the previous one did not: Recall@50 must come out
    identical between them, and a difference means the harness is wrong.
    """
    app_ids = retrieve_rrf(engine, query)
    if not app_ids:
        return []

    records_map = engine._fetch_game_details(app_ids)
    fused = dict(engine._rrf_fuse([app_ids]))
    ordered = []
    for app_id in app_ids:
        rec = records_map.get(app_id)
        if rec is not None:
            rec.raw["_rrf"] = fused.get(app_id, 0.0)
            ordered.append(rec)

    engine._assign_retrieval_scores(ordered)
    ranked = engine.rank_candidates(query, ordered)
    ranked_ids = [rec.app_id for rec, _ in ranked]

    # rank_candidates() returns only the shortlist; keep the rest of the
    # candidate set behind it so recall at depth stays comparable.
    tail = [a for a in app_ids if a not in set(ranked_ids)]
    return ranked_ids + tail


CONFIGS = {
    "bm25": retrieve_bm25,
    "vector": retrieve_vector,
    "rrf": retrieve_rrf,
    "full": retrieve_full,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def load_queries(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def judgements(record: dict) -> tuple[set[str], dict[str, float]]:
    """Return (relevant ids, graded relevance) for a query record.

    Known-item records name a single correct game; topical records carry a
    grade map produced by tag overlap.
    """
    if "gold_appid" in record:
        gold = str(record["gold_appid"])
        return {gold}, {gold: 2.0}
    grades = {str(k): float(v) for k, v in record.get("grades", {}).items() if v > 0}
    return set(grades), grades


def evaluate(engine, queries: list[dict], retrieve, verbose: bool) -> dict[str, list[float]]:
    per_query: dict[str, list[float]] = {
        "Recall@50": [], "nDCG@10": [], "MRR": [], "P@5": [],
    }
    started = time.perf_counter()
    for i, record in enumerate(queries, 1):
        relevant, grades = judgements(record)
        try:
            retrieved = retrieve(engine, record["query"])
        except Exception as exc:  # a failing query is a zero, not a crashed run
            print(f"  ! query {record.get('query_id')} failed: {exc}")
            retrieved = []

        per_query["Recall@50"].append(M.recall_at_k(retrieved, relevant, 50))
        per_query["nDCG@10"].append(M.ndcg_at_k(retrieved, grades, 10))
        per_query["MRR"].append(M.reciprocal_rank(retrieved, relevant))
        per_query["P@5"].append(M.precision_at_k(retrieved, relevant, 5))

        if verbose and i % 50 == 0:
            rate = i / (time.perf_counter() - started)
            print(f"  {i}/{len(queries)}  ({rate:.1f} q/s)")
    return per_query


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the retrieval ablation")
    ap.add_argument("--queries", type=Path,
                    default=Path(__file__).resolve().parent / "queries_knownitem.jsonl")
    ap.add_argument("--configs", default="bm25,vector,rrf,full",
                    help="comma-separated subset of " + ",".join(CONFIGS))
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N queries")
    ap.add_argument("--label", default="", help="suffix appended to each configuration label")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    names = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in names if c not in CONFIGS]
    if unknown:
        sys.exit(f"Unknown configuration(s): {', '.join(unknown)}")

    queries = load_queries(args.queries)
    if args.limit:
        queries = queries[: args.limit]
    print(f"Queries: {len(queries)} from {args.queries.name}")
    print(f"Index:   {os.environ.get('RAGLOOKER_INDEX_DIR', 'vector_index (default)')}")

    from recommender import GameSearchEngine, DB_PATH
    engine = GameSearchEngine(DB_PATH)

    rows = []
    for name in names:
        label = f"{name}{(' ' + args.label) if args.label else ''}"
        print(f"\n--- {label} ---")
        started = time.perf_counter()
        per_query = evaluate(engine, queries, CONFIGS[name], not args.quiet)
        stats = M.summarise(per_query)
        rows.append((label, stats))
        print(f"  done in {time.perf_counter() - started:.0f}s")
        for metric, (mean, se) in stats.items():
            print(f"    {metric:10s} {mean:.3f} ± {se:.3f}")

    table = M.to_markdown(rows, ["Recall@50", "nDCG@10", "MRR", "P@5"])
    print("\n" + table)

    RESULTS_DIR.mkdir(exist_ok=True)
    out = args.out or RESULTS_DIR / f"ablation_{int(time.time())}.md"
    out.write_text(
        f"# Ablation results\n\n"
        f"- Queries: {len(queries)} from `{args.queries.name}`\n"
        f"- Index: `{os.environ.get('RAGLOOKER_INDEX_DIR', 'vector_index')}`\n\n"
        f"{table}\n"
    )
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
