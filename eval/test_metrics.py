"""Unit tests for eval/metrics.py (PRD-002, acceptance criterion A1).

Runs without the database, the FAISS index, or Ollama:

    uv run python eval/test_metrics.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from metrics import (  # noqa: E402
    dedupe,
    recall_at_k,
    precision_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
    dcg_at_k,
    ndcg_at_k,
    jaccard,
    intra_list_diversity,
    summarise,
    to_markdown,
)

passed = failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"✅ {name}")
    else:
        failed += 1
        print(f"❌ {name}  {detail}")


def close(a, b, tol=1e-12):
    return abs(a - b) < tol


# ---------------------------------------------------------------------------
print("\n--- recall@k ---")

check("all relevant retrieved", close(recall_at_k(["a", "b", "c"], ["a", "b"], 3), 1.0))
check("half retrieved", close(recall_at_k(["a", "x", "y"], ["a", "b"], 3), 0.5))
check("k truncates before the hit", close(recall_at_k(["x", "a"], ["a"], 1), 0.0))
check("hit exactly at k", close(recall_at_k(["x", "a"], ["a"], 2), 1.0))
check("no relevant items returns 0.0", close(recall_at_k(["a"], [], 5), 0.0))
check("empty retrieved returns 0.0", close(recall_at_k([], ["a"], 5), 0.0))
check("k larger than result list", close(recall_at_k(["a"], ["a", "b"], 50), 0.5))
check("k=0 returns 0.0", close(recall_at_k(["a"], ["a"], 0), 0.0))
check("duplicates do not inflate recall",
      close(recall_at_k(["a", "a", "a"], ["a", "b"], 3), 0.5),
      recall_at_k(["a", "a", "a"], ["a", "b"], 3))
check("duplicates do not inflate precision",
      close(precision_at_k(["a", "a"], ["a"], 2), 0.5))
check("duplicates do not inflate DCG",
      close(dcg_at_k(["a", "a"], {"a": 1.0}, 2), 1.0))
check("dedupe keeps first position",
      dedupe(["b", "a", "b", "c"]) == ["b", "a", "c"])
check("dedupe of empty list", dedupe([]) == [])

# ---------------------------------------------------------------------------
print("\n--- precision@k ---")

check("3 of 5 relevant", close(precision_at_k(["a", "b", "c", "x", "y"], ["a", "b", "c"], 5), 0.6))
check("perfect precision", close(precision_at_k(["a", "b"], ["a", "b"], 2), 1.0))
check("denominator is k, not len(retrieved)",
      close(precision_at_k(["a"], ["a"], 5), 0.2),
      "short result lists must be penalised")
check("k=0 returns 0.0", close(precision_at_k(["a"], ["a"], 0), 0.0))
check("no relevant returns 0.0", close(precision_at_k(["a", "b"], [], 2), 0.0))

# ---------------------------------------------------------------------------
print("\n--- reciprocal rank ---")

check("first position", close(reciprocal_rank(["a", "b"], ["a"]), 1.0))
check("second position", close(reciprocal_rank(["x", "a"], ["a"]), 0.5))
check("tenth position", close(reciprocal_rank(list("bcdefghij") + ["a"], ["a"]), 0.1))
check("not retrieved", close(reciprocal_rank(["x", "y"], ["a"]), 0.0))
check("uses the earliest relevant item",
      close(reciprocal_rank(["x", "b", "a"], ["a", "b"]), 0.5))
check("empty retrieved", close(reciprocal_rank([], ["a"]), 0.0))

check("MRR averages over queries",
      close(mean_reciprocal_rank([(["a"], ["a"]), (["x", "a"], ["a"])]), 0.75))
check("MRR of empty batch", close(mean_reciprocal_rank([]), 0.0))

# ---------------------------------------------------------------------------
print("\n--- DCG / nDCG ---")

# Position discounts: log2(2)=1, log2(3)≈1.585, log2(4)=2
check("DCG applies positional discount",
      close(dcg_at_k(["a", "b"], {"a": 2.0, "b": 2.0}, 2), 2.0 + 2.0 / math.log2(3)))
check("DCG ignores irrelevant items",
      close(dcg_at_k(["x", "a"], {"a": 1.0}, 2), 1.0 / math.log2(3)))
check("DCG respects k", close(dcg_at_k(["x", "a"], {"a": 1.0}, 1), 0.0))

check("perfect ranking scores 1.0",
      close(ndcg_at_k(["a", "b"], {"a": 2.0, "b": 1.0}, 2), 1.0))
check("swapped ranking scores below 1.0",
      ndcg_at_k(["b", "a"], {"a": 2.0, "b": 1.0}, 2) < 1.0)
check("nDCG within [0,1]",
      0.0 <= ndcg_at_k(["b", "a", "x"], {"a": 2.0, "b": 1.0}, 3) <= 1.0)
check("no relevant items returns 0.0", close(ndcg_at_k(["a"], {}, 5), 0.0))
check("all-zero grades return 0.0", close(ndcg_at_k(["a"], {"a": 0.0}, 5), 0.0))
check("k=0 returns 0.0", close(ndcg_at_k(["a"], {"a": 2.0}, 0), 0.0))
check("empty retrieved returns 0.0", close(ndcg_at_k([], {"a": 2.0}, 5), 0.0))

# An unretrieved relevant item must still count in the ideal ranking, otherwise
# missing a document would be free.
partial = ndcg_at_k(["a"], {"a": 2.0, "b": 2.0}, 2)
check("unretrieved relevant items stay in IDCG",
      partial < 1.0, f"got {partial:.4f}, expected < 1.0")

check("higher grade outranking lower is better",
      ndcg_at_k(["a", "b"], {"a": 2.0, "b": 1.0}, 2)
      > ndcg_at_k(["b", "a"], {"a": 2.0, "b": 1.0}, 2))

# ---------------------------------------------------------------------------
print("\n--- diversity ---")

check("identical sets are fully similar", close(jaccard(["a"], ["a"]), 1.0))
check("disjoint sets share nothing", close(jaccard(["a"], ["b"]), 0.0))
check("half overlap", close(jaccard(["a", "b"], ["b", "c"]), 1 / 3))
check("two empty sets return 0.0", close(jaccard([], []), 0.0))

check("identical tag sets have zero diversity",
      close(intra_list_diversity([["rpg"], ["rpg"], ["rpg"]]), 0.0))
check("disjoint tag sets have full diversity",
      close(intra_list_diversity([["a"], ["b"], ["c"]]), 1.0))
check("partial overlap sits in between",
      0.0 < intra_list_diversity([["a", "b"], ["b", "c"], ["x"]]) < 1.0)
check("single item returns 0.0", close(intra_list_diversity([["a"]]), 0.0))
check("empty list returns 0.0", close(intra_list_diversity([]), 0.0))

# ---------------------------------------------------------------------------
print("\n--- summarise ---")

s = summarise({"recall": [1.0, 0.0, 1.0, 0.0]})
check("mean is correct", close(s["recall"][0], 0.5), s["recall"])
check("standard error is positive for varied input", s["recall"][1] > 0)

s = summarise({"recall": [0.5, 0.5, 0.5]})
check("identical values give zero standard error", close(s["recall"][1], 0.0))

s = summarise({"recall": [0.7]})
check("single value: mean is itself, se 0.0", close(s["recall"][0], 0.7) and close(s["recall"][1], 0.0))

s = summarise({"recall": []})
check("empty list returns (0.0, 0.0)", s["recall"] == (0.0, 0.0))

# Standard error must shrink as the sample grows -- this is why 500 queries
# were specified rather than 50.
se_small = summarise({"m": [1.0, 0.0] * 10})["m"][1]
se_large = summarise({"m": [1.0, 0.0] * 100})["m"][1]
check("standard error shrinks with sample size",
      se_large < se_small, f"{se_large:.4f} vs {se_small:.4f}")

# ---------------------------------------------------------------------------
print("\n--- markdown output ---")

table = to_markdown(
    [("BM25 only", summarise({"Recall@50": [0.5, 0.7], "nDCG@10": [0.4, 0.4]}))],
    ["Recall@50", "nDCG@10"],
)
check("table has header, separator and one row", len(table.splitlines()) == 3, table)
check("row label present", "BM25 only" in table)
check("values rendered with uncertainty", "±" in table)

missing = to_markdown([("x", {})], ["Recall@50"])
check("absent metric renders as zero", "0.000 ± 0.000" in missing, missing)

print(f"\n{passed}/{passed + failed} passed")
raise SystemExit(1 if failed else 0)
