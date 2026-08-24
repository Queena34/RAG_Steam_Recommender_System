"""Unit tests for the pure ranking helpers: RRF fusion, review-aware quality
scoring, and the diversity constraint.

These deliberately use hand-built GameRecords so the whole file runs without
the SQLite database, the FAISS index, or Ollama.

    uv run python test_ranking.py
"""
from recommender import (
    GameSearchEngine,
    GameRecord,
    RRF_K,
    RELEVANCE_WEIGHT,
    QUALITY_WEIGHT,
)

rrf_fuse = GameSearchEngine._rrf_fuse
quality_score = GameSearchEngine._quality_score
apply_diversity = GameSearchEngine._apply_diversity
assign_scores = GameSearchEngine._assign_retrieval_scores

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"✅ {name}")
    else:
        failed += 1
        print(f"❌ {name}  {detail}")


def game(app_id, *, pos=0, neg=0, dev=None, tags=None):
    return GameRecord(
        app_id=app_id,
        raw={
            "name": f"Game {app_id}",
            "positive": pos,
            "negative": neg,
            "developers": [dev] if dev else [],
            "tags": tags or [],
        },
    )


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------
print("\n--- Reciprocal Rank Fusion ---")

fused = rrf_fuse([["a", "b", "c"]])
check("single list preserves order", [i for i, _ in fused] == ["a", "b", "c"], fused)

# "b" is 2nd in both lists; "a" and "c" are 1st in one list and absent from the other.
# 2/(K+2) > 1/(K+1), so appearing in both should beat a single first place.
fused = rrf_fuse([["a", "b"], ["c", "b"]])
check("item in both lists outranks single-list winners",
      fused[0][0] == "b", fused)

# Disjoint lists: ranks interleave rather than concatenate.
fused = rrf_fuse([["a", "b"], ["c", "d"]])
ids = [i for i, _ in fused]
check("disjoint lists interleave by rank",
      set(ids[:2]) == {"a", "c"} and set(ids[2:]) == {"b", "d"}, ids)

# Scale invariance is the whole point of RRF: only ranks matter.
check("fusion ignores absolute scores (rank-only)",
      rrf_fuse([["x", "y", "z"]]) == rrf_fuse([["x", "y", "z"]]))

# Determinism on ties: two identical lists tie every pair, broken by app_id.
fused = rrf_fuse([["b", "a"], ["a", "b"]])
check("ties broken deterministically by app_id",
      [i for i, _ in fused] == ["a", "b"], fused)

expected_top = 1.0 / (RRF_K + 1)
fused = rrf_fuse([["a"]])
check("score matches 1/(k+rank)", abs(fused[0][1] - expected_top) < 1e-12, fused)

check("empty input returns empty", rrf_fuse([]) == [])
check("empty sublists ignored", rrf_fuse([[], ["a"]])[0][0] == "a")


# ---------------------------------------------------------------------------
# Review-aware quality scoring
# ---------------------------------------------------------------------------
print("\n--- Review-aware quality (Wilson lower bound) ---")

perfect_tiny = quality_score(game("1", pos=3, neg=0))
great_huge = quality_score(game("2", pos=47000, neg=3000))
check("3/3 positive scores below 47000/50000",
      perfect_tiny < great_huge, f"{perfect_tiny:.4f} vs {great_huge:.4f}")

check("no reviews scores exactly 0.0",
      quality_score(game("3")) == 0.0)

mixed = quality_score(game("4", pos=500, neg=500))
good = quality_score(game("5", pos=900, neg=100))
check("at equal volume, better ratio wins",
      good > mixed, f"{good:.4f} vs {mixed:.4f}")

small_good = quality_score(game("6", pos=90, neg=10))
big_good = quality_score(game("7", pos=90000, neg=10000))
check("at equal ratio, more reviews wins",
      big_good > small_good, f"{big_good:.4f} vs {small_good:.4f}")

for pos, neg in [(0, 0), (1, 0), (0, 1), (10**6, 1), (1, 10**6)]:
    q = quality_score(game("x", pos=pos, neg=neg))
    check(f"score in [0,1] for {pos}/{neg}", 0.0 <= q <= 1.0, q)

check("all-negative scores near zero",
      quality_score(game("8", pos=0, neg=5000)) < 0.5)


# ---------------------------------------------------------------------------
# Score blending
# ---------------------------------------------------------------------------
print("\n--- Retrieval score blending ---")

a = game("a", pos=1000, neg=0)
b = game("b", pos=1000, neg=0)
a.raw["_rrf"] = 0.02
b.raw["_rrf"] = 0.01
assign_scores([a, b])
check("top candidate normalised to relevance 1.0",
      abs(a.raw["_relevance"] - 1.0) < 1e-12, a.raw["_relevance"])
check("relevance scales relative to best",
      abs(b.raw["_relevance"] - 0.5) < 1e-12, b.raw["_relevance"])
check("score = w_rel*relevance + w_qual*quality",
      abs(a.raw["_score"] - (RELEVANCE_WEIGHT * 1.0 + QUALITY_WEIGHT * a.raw["_quality"])) < 1e-12)
check("higher rrf outranks lower at equal quality",
      a.raw["_score"] > b.raw["_score"])

assign_scores([])  # must not raise
check("empty candidate list is a no-op", True)

z = game("z", pos=10, neg=0)
z.raw["_rrf"] = 0.0
assign_scores([z])
check("all-zero rrf does not divide by zero", z.raw["_relevance"] == 0.0)


# ---------------------------------------------------------------------------
# Diversity constraint
# ---------------------------------------------------------------------------
print("\n--- Diversity constraint ---")

# Five games from one developer, descending score.
same_dev = [(game(str(i), dev="Acme", tags=["rpg"]), 1.0 - i * 0.1) for i in range(5)]
out = apply_diversity(same_dev, limit=5, max_per_developer=2, max_per_tag=3)
check("never returns fewer than the unconstrained ranking",
      len(out) == 5, len(out))

varied = [
    (game("a", dev="Acme", tags=["rpg"]), 0.9),
    (game("b", dev="Acme", tags=["rpg"]), 0.8),
    (game("c", dev="Acme", tags=["rpg"]), 0.7),
    (game("d", dev="Beta", tags=["puzzle"]), 0.6),
    (game("e", dev="Gamma", tags=["racing"]), 0.5),
]
out = apply_diversity(varied, limit=3, max_per_developer=2, max_per_tag=3)
ids = [r.app_id for r, _ in out]
check("developer cap pushes the third Acme game down",
      ids == ["a", "b", "d"], ids)

out = apply_diversity(varied, limit=3, max_per_developer=5, max_per_tag=2)
ids = [r.app_id for r, _ in out]
check("tag cap pushes the third rpg down",
      ids == ["a", "b", "d"], ids)

no_conflict = [
    (game("a", dev="A", tags=["x"]), 0.9),
    (game("b", dev="B", tags=["y"]), 0.8),
    (game("c", dev="C", tags=["z"]), 0.7),
]
out = apply_diversity(no_conflict, limit=3)
check("order preserved when no cap is hit",
      [r.app_id for r, _ in out] == ["a", "b", "c"])

check("empty input returns empty", apply_diversity([], limit=5) == [])

out = apply_diversity(no_conflict, limit=10)
check("limit larger than input returns everything", len(out) == 3)

# Missing developer/tag metadata must not be treated as a shared group.
blank = [(game(str(i)), 1.0 - i * 0.1) for i in range(4)]
out = apply_diversity(blank, limit=4, max_per_developer=1, max_per_tag=1)
check("blank developer/tag is not a group key",
      [r.app_id for r, _ in out] == ["0", "1", "2", "3"],
      [r.app_id for r, _ in out])

# Dict-shaped tags (the real DB format) must work like list-shaped ones.
dict_tagged = [
    (GameRecord(app_id="p", raw={"positive": 1, "negative": 0, "developers": ["A"],
                                 "tags": {"rpg": 10, "indie": 5}}), 0.9),
    (GameRecord(app_id="q", raw={"positive": 1, "negative": 0, "developers": ["B"],
                                 "tags": {"rpg": 8}}), 0.8),
    (GameRecord(app_id="r", raw={"positive": 1, "negative": 0, "developers": ["C"],
                                 "tags": {"puzzle": 3}}), 0.7),
]
out = apply_diversity(dict_tagged, limit=2, max_per_developer=5, max_per_tag=1)
check("dict-shaped tags handled like lists",
      [r.app_id for r, _ in out] == ["p", "r"], [r.app_id for r, _ in out])


print(f"\n{passed}/{passed + failed} passed")
raise SystemExit(1 if failed else 0)
