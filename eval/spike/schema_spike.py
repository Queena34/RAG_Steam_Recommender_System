#!/usr/bin/env python3
"""PRD-004 step 0: does phi4-mini hold up under JSON-schema constrained decoding?

Nothing is changed in the production path here. The point is to find out, before
writing any of it, whether a 3.8B model on CPU returns schema-valid output often
enough for the approach to be worth building, and what it costs in latency.

Measured per query:
  * valid JSON returned under the schema
  * exactly five recommendations
  * how many app_ids fall outside the candidate set (entity hallucination)
  * latency

    uv run python eval/spike/schema_spike.py --n 20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from recommender import GameSearchEngine, DB_PATH, DEFAULT_MATCH_COUNT  # noqa: E402

RECOMMENDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "recommendations": {
            "type": "array",
            "minItems": DEFAULT_MATCH_COUNT,
            "maxItems": DEFAULT_MATCH_COUNT,
            "items": {
                "type": "object",
                "properties": {
                    "app_id": {"type": "string"},
                    "reason": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["app_id", "reason", "evidence"],
            },
        }
    },
    "required": ["recommendations"],
}


def build_prompt(query: str, candidates) -> str:
    """Candidate list keyed by app_id, so the model returns an identifier.

    Kept deliberately short: prompt prefill dominates latency on CPU, and the
    schema already enforces the output shape that the current implementation
    has to spell out in three capitalised sentences.
    """
    lines = []
    for rec in candidates:
        tags = ", ".join(rec._normalize_tags(rec.raw.get("tags"))[:5])
        pos = rec.raw.get("positive") or 0
        neg = rec.raw.get("negative") or 0
        rating = f"{round(100 * pos / (pos + neg))}% positive" if pos + neg else "unrated"
        desc = rec.short_description[:120]
        lines.append(f'{rec.app_id}: {rec.name} [{tags}] ({rating}) {desc}')

    return (
        f'A player asks for: "{query}"\n\n'
        f"Candidate games, one per line as app_id: name [tags] (rating) description\n\n"
        + "\n".join(lines)
        + f"\n\nPick the {DEFAULT_MATCH_COUNT} best matches. For each, give the app_id "
        f"exactly as listed, a reason it fits this request, and the evidence you "
        f"used -- a tag, the rating, or wording from the description."
    )


def run_one(engine, query: str, n_candidates: int = 10) -> dict:
    import ollama

    candidates = engine.retrieve_candidates(query)
    ranked = engine.rank_candidates(query, candidates)[:n_candidates]
    records = [rec for rec, _ in ranked]
    valid_ids = {rec.app_id for rec in records}
    if not records:
        return {"query": query, "error": "no candidates"}

    prompt = build_prompt(query, records)
    started = time.perf_counter()
    try:
        resp = ollama.generate(
            model=engine.llm_model,
            prompt=prompt,
            format=RECOMMENDATION_SCHEMA,
            options={"temperature": 0.3, "num_predict": 900},
        )
        raw = resp.response
    except Exception as exc:
        return {"query": query, "error": f"generate failed: {exc}",
                "latency": time.perf_counter() - started}
    latency = time.perf_counter() - started

    out = {"query": query, "latency": latency, "n_candidates": len(records),
           "raw_len": len(raw)}
    try:
        data = json.loads(raw)
    except Exception as exc:
        out.update(valid_json=False, error=f"json: {exc}", raw_head=raw[:200])
        return out

    recs = data.get("recommendations")
    if not isinstance(recs, list):
        out.update(valid_json=False, error="no recommendations array")
        return out

    ids = [r.get("app_id") for r in recs if isinstance(r, dict)]
    outside = [i for i in ids if i not in valid_ids]
    out.update(
        valid_json=True,
        n_returned=len(recs),
        exactly_five=len(recs) == DEFAULT_MATCH_COUNT,
        n_outside_candidates=len(outside),
        outside_ids=outside[:5],
        has_all_fields=all(
            isinstance(r, dict) and {"app_id", "reason", "evidence"} <= set(r)
            for r in recs
        ),
        sample=recs[0] if recs else None,
    )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--queries", type=Path,
                    default=BASE_DIR / "eval" / "queries_topical.jsonl")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "schema_spike_results.jsonl")
    args = ap.parse_args()

    queries = [json.loads(l)["query"] for l in args.queries.read_text().splitlines() if l.strip()]
    queries = queries[: args.n]

    engine = GameSearchEngine(DB_PATH)
    if not engine.llm_model:
        sys.exit("No LLM available; cannot run the spike.")
    print(f"\nModel: {engine.llm_model}   queries: {len(queries)}\n")

    results = []
    for i, q in enumerate(queries, 1):
        r = run_one(engine, q)
        results.append(r)
        status = "ok " if r.get("valid_json") else "FAIL"
        extra = ""
        if r.get("valid_json"):
            extra = (f"n={r['n_returned']} outside={r['n_outside_candidates']}"
                     f" fields={'y' if r['has_all_fields'] else 'n'}")
        else:
            extra = r.get("error", "")[:60]
        print(f"  [{i:2d}/{len(queries)}] {r.get('latency', 0):6.1f}s  {status}  {extra}")

    args.out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n")

    ok = [r for r in results if r.get("valid_json")]
    five = [r for r in ok if r.get("exactly_five")]
    fields = [r for r in ok if r.get("has_all_fields")]
    clean = [r for r in ok if r.get("n_outside_candidates") == 0]
    lat = [r["latency"] for r in results if "latency" in r]
    n = len(results)
    print(f"\n{'='*52}")
    print(f"  valid JSON            {len(ok)}/{n} ({100*len(ok)/n:.0f}%)   threshold 80%")
    print(f"  exactly five items    {len(five)}/{n} ({100*len(five)/n:.0f}%)")
    print(f"  all fields present    {len(fields)}/{n} ({100*len(fields)/n:.0f}%)")
    print(f"  no ids outside set    {len(clean)}/{n} ({100*len(clean)/n:.0f}%)")
    if lat:
        lat_sorted = sorted(lat)
        print(f"  latency  median {lat_sorted[len(lat)//2]:.1f}s"
              f"  min {lat_sorted[0]:.1f}s  max {lat_sorted[-1]:.1f}s")
    print(f"\n  verdict: {'PASS' if len(ok)/n >= 0.8 else 'FAIL'} (>= 80% valid JSON)")
    print(f"  written to {args.out}")


if __name__ == "__main__":
    main()
