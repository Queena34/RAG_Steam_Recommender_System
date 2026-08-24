"""Retrieval metrics for the ablation harness (PRD-002, R1 and R2).

Every function here is pure: it takes ranked ids and relevance judgements and
returns a number. Nothing touches the database, the index, or Ollama, so the
whole module is testable with hand-built data (see test_metrics.py).

Two conventions are fixed here and should not be changed silently, because
doing so would make previously reported numbers incomparable:

* **Gain is linear.** A relevance grade of 2 contributes a gain of 2, matching
  OPTIMIZED_DESIGN.md section 10.3. The exponential convention (2^rel - 1) is
  equally standard and would produce different -- not wrong -- numbers.
* **Undefined cases return 0.0** rather than raising or returning NaN. A query
  with no relevant items has no meaningful recall; scoring it 0.0 keeps batch
  averages simple at the cost of slightly depressing them. Filter such queries
  out of the evaluation set rather than relying on this behaviour.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dedupe(retrieved: Sequence[str]) -> list[str]:
    """Drop repeat ids, keeping the first (best) position of each.

    A ranking is a list of distinct documents. Without this, an item appearing
    twice would be counted twice: recall could exceed 1.0, and a buggy fusion
    that emitted duplicates would score *better* rather than being caught.
    Every metric below deduplicates on entry.
    """
    seen: set[str] = set()
    out: list[str] = []
    for app_id in retrieved:
        if app_id not in seen:
            seen.add(app_id)
            out.append(app_id)
    return out


# ---------------------------------------------------------------------------
# Set-based metrics
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Share of the relevant items that appear in the top k.

    This is the metric for the *retrieval* stage. An item missing from the top
    k cannot be recovered by any amount of re-ranking, so recall is the ceiling
    on everything downstream.
    """
    relevant = set(relevant)
    if not relevant or k <= 0:
        return 0.0
    hits = sum(1 for app_id in dedupe(retrieved)[:k] if app_id in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Share of the top k that is relevant.

    Divided by k rather than by the number of results returned, so a system
    that returns fewer than k items is penalised for the shortfall.
    """
    relevant = set(relevant)
    if k <= 0:
        return 0.0
    hits = sum(1 for app_id in dedupe(retrieved)[:k] if app_id in relevant)
    return hits / k


# ---------------------------------------------------------------------------
# Rank-based metrics
# ---------------------------------------------------------------------------

def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """1 / rank of the first relevant item; 0.0 if none is retrieved.

    Extremely sensitive to the head of the ranking -- dropping from position 1
    to position 2 halves the score -- which mirrors how far users actually
    read. Used as the headline metric for known-item retrieval, where there is
    exactly one correct answer.
    """
    relevant = set(relevant)
    for rank, app_id in enumerate(dedupe(retrieved), 1):
        if app_id in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    runs: Iterable[tuple[Sequence[str], Iterable[str]]]
) -> float:
    """Mean of reciprocal_rank over (retrieved, relevant) pairs."""
    scores = [reciprocal_rank(r, rel) for r, rel in runs]
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Graded metrics
# ---------------------------------------------------------------------------

def dcg_at_k(retrieved: Sequence[str], grades: Mapping[str, float], k: int) -> float:
    """Discounted cumulative gain over the top k.

    Each position contributes gain / log2(position + 1), so the same item is
    worth less the further down it sits: position 1 keeps its full gain,
    position 3 keeps half, position 10 keeps under a third.
    """
    total = 0.0
    for rank, app_id in enumerate(dedupe(retrieved)[:k], 1):
        gain = grades.get(app_id, 0.0)
        if gain:
            total += gain / math.log2(rank + 1)
    return total


def ndcg_at_k(retrieved: Sequence[str], grades: Mapping[str, float], k: int) -> float:
    """DCG normalised by the best achievable DCG for this query.

    The ideal ranking is built from *all* known relevant items, not just the
    retrieved ones, so failing to retrieve a relevant item is penalised rather
    than silently excluded from the denominator.

    Returns a value in [0, 1]; 1.0 means the ranking is optimal.
    """
    if k <= 0:
        return 0.0
    ideal_gains = sorted((g for g in grades.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal_gains, 1))
    if idcg <= 0:
        return 0.0
    return dcg_at_k(retrieved, grades, k) / idcg


# ---------------------------------------------------------------------------
# List-level metrics
# ---------------------------------------------------------------------------

def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """|A n B| / |A u B|; 0.0 when both sets are empty."""
    a, b = set(a), set(b)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def intra_list_diversity(tag_sets: Sequence[Iterable[str]]) -> float:
    """1 - mean pairwise Jaccard similarity of the items' tag sets.

    Measures whether a shortlist covers different kinds of game or five
    variations on one. Used to verify that the diversity constraint (C6) has a
    measurable effect: relevance metrics alone cannot distinguish a good
    shortlist from five entries of the same series.

    Fewer than two items has no pairwise similarity to average, so it returns
    0.0. In practice lists are always of length 5.
    """
    sets = [set(t) for t in tag_sets]
    if len(sets) < 2:
        return 0.0
    sims = [
        jaccard(sets[i], sets[j])
        for i in range(len(sets))
        for j in range(i + 1, len(sets))
    ]
    return 1.0 - sum(sims) / len(sims)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def summarise(
    per_query: Mapping[str, Sequence[float]]
) -> dict[str, tuple[float, float]]:
    """Reduce {metric: [per-query values]} to {metric: (mean, std_error)}.

    The standard error is reported alongside the mean so that a difference
    between two ablation rows can be checked against measurement noise instead
    of being read as an improvement on sight.
    """
    out: dict[str, tuple[float, float]] = {}
    for name, values in per_query.items():
        n = len(values)
        if n == 0:
            out[name] = (0.0, 0.0)
            continue
        mean = sum(values) / n
        if n == 1:
            out[name] = (mean, 0.0)
            continue
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        out[name] = (mean, math.sqrt(variance / n))
    return out


def to_markdown(
    rows: Sequence[tuple[str, Mapping[str, tuple[float, float]]]],
    metrics: Sequence[str],
    decimals: int = 3,
) -> str:
    """Render ablation rows as a Markdown table ready to paste into the report.

    `rows` is a sequence of (configuration label, summarise() output).
    """
    header = "| Configuration | " + " | ".join(metrics) + " |"
    sep = "|---" * (len(metrics) + 1) + "|"
    lines = [header, sep]
    for label, stats in rows:
        cells = []
        for m in metrics:
            mean, se = stats.get(m, (0.0, 0.0))
            cells.append(f"{mean:.{decimals}f} ± {se:.{decimals}f}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
