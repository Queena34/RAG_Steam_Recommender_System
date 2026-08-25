"""Constructed tests for the structured generation path (PRD-004, A2-A6).

Each case feeds a fabricated model response into the engine, so the failure
modes that matter can be produced on demand rather than waited for. Nothing
here calls Ollama, the FAISS index, or the database.

    uv run python test_generation.py
"""
from __future__ import annotations

from recommender import (
    GameSearchEngine,
    GameRecord,
    DEFAULT_MATCH_COUNT,
    GEN_STRUCTURED,
    GEN_RETRY,
    GEN_PARTIAL,
    GEN_FALLBACK,
    RECOMMENDATION_SCHEMA,
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


def game(app_id, name=None, tags=None, desc="", pos=1000, neg=10, genres=None):
    return GameRecord(
        app_id=app_id,
        raw={
            "name": name or f"Game {app_id}",
            "short_description": desc,
            "tags": tags or ["indie"],
            "genres": genres or [],
            "positive": pos,
            "negative": neg,
            "review_summary": "",
        },
    )


def engine_with(responses):
    """An engine whose schema calls return `responses` in order.

    Bypasses __init__ so no index, database, or model is touched.
    """
    e = object.__new__(GameSearchEngine)
    e.llm_model = "stub"
    calls = {"n": 0}

    def _call_schema(prompt, temperature):
        i = calls["n"]
        calls["n"] += 1
        return responses[i] if i < len(responses) else None

    e._call_schema = _call_schema
    e._calls = calls
    return e


def matches(n=10):
    return [(game(str(100 + i), name=f"Title {i}"), 1.0 - i * 0.05) for i in range(n)]


def rec(app_id, reason="fits", evidence="indie"):
    return {"app_id": app_id, "reason": reason, "evidence": evidence}


# ---------------------------------------------------------------------------
print("\n--- schema shape ---")

items = RECOMMENDATION_SCHEMA["properties"]["recommendations"]
check("array is pinned to exactly five entries",
      items["minItems"] == items["maxItems"] == DEFAULT_MATCH_COUNT)
check("app_id, reason and evidence are all required",
      set(items["items"]["required"]) == {"app_id", "reason", "evidence"})
check("app_id is typed as a string",
      items["items"]["properties"]["app_id"]["type"] == "string")

# ---------------------------------------------------------------------------
print("\n--- happy path ---")

m = matches()
ok = {"recommendations": [rec(str(100 + i)) for i in range(5)]}
answer, ids, tele = engine_with([ok]).generate_answer("q", m)
check("five identifiers returned", len(ids) == 5, ids)
check("order follows the model's array", ids == ["100", "101", "102", "103", "104"], ids)
check("mode is structured", tele["generation_mode"] == GEN_STRUCTURED, tele)
check("nothing rejected", tele["rejected_app_ids"] == 0)
check("no retry needed", tele["retries"] == 0)
check("answer names the games", "Title 0" in answer, answer[:80])

# A2 --------------------------------------------------------------------
print("\n--- A2: identifiers outside the candidate set ---")

hallucinated = {"recommendations": [
    rec("100"), rec("999999"), rec("101"), rec("abc"), rec("102"),
]}
answer, ids, tele = engine_with([hallucinated]).generate_answer("q", matches())
check("invented identifiers are dropped",
      "999999" not in ids and "abc" not in ids, ids)
check("valid identifiers survive", ids == ["100", "101", "102"], ids)
check("rejections are counted", tele["rejected_app_ids"] == 2, tele)
check("shortfall is reported as partial",
      tele["generation_mode"] == GEN_PARTIAL, tele)

# Every entry outside the set: nothing usable is left.
allbad = {"recommendations": [rec(str(900000 + i)) for i in range(5)]}
answer, ids, tele = engine_with([allbad]).generate_answer("q", matches())
check("all-invalid output falls back", tele["generation_mode"] == GEN_FALLBACK, tele)
check("no identifiers returned when all are invalid", ids == [], ids)

# Duplicates must not fill the shortlist with one game repeated.
dupes = {"recommendations": [rec("100")] * 5}
answer, ids, tele = engine_with([dupes]).generate_answer("q", matches())
check("repeated identifiers are collapsed", ids == ["100"], ids)

# A3 --------------------------------------------------------------------
print("\n--- A3: retry on unusable output ---")

e = engine_with([None, ok])
answer, ids, tele = e.generate_answer("q", matches())
check("a second attempt is made", e._calls["n"] == 2, e._calls)
check("retry is recorded", tele["retries"] == 1, tele)
check("mode marks the retry", tele["generation_mode"] == GEN_RETRY, tele)
check("recovered output is used", len(ids) == 5, ids)

e = engine_with([None, None])
answer, ids, tele = e.generate_answer("q", matches())
check("only one retry is attempted", e._calls["n"] == 2, e._calls)
check("two failures fall back", tele["generation_mode"] == GEN_FALLBACK, tele)

# A4 --------------------------------------------------------------------
print("\n--- A4: partial output ---")

short = {"recommendations": [rec("100"), rec("101")]}
answer, ids, tele = engine_with([short]).generate_answer("q", matches())
check("mode is partial", tele["generation_mode"] == GEN_PARTIAL, tele)
check("the model's picks are kept", ids == ["100", "101"], ids)

# A5 --------------------------------------------------------------------
print("\n--- A5: degradation is never silent ---")

for label, resp in [("empty array", {"recommendations": []}),
                    ("missing key", {"nope": []}),
                    ("null", None)]:
    answer, ids, tele = engine_with([resp, resp]).generate_answer("q", matches())
    check(f"{label} falls back without raising",
          tele["generation_mode"] == GEN_FALLBACK and isinstance(answer, str), tele)

answer, ids, tele = engine_with([ok]).generate_answer("q", [])
check("no candidates returns a usable answer",
      ids == [] and tele["generation_mode"] == GEN_FALLBACK and answer)

e = object.__new__(GameSearchEngine)
e.llm_model = None
answer, ids, tele = e.generate_answer("q", matches())
check("no model available falls back",
      tele["generation_mode"] == GEN_FALLBACK and isinstance(answer, str))

check("every mode constant is distinct",
      len({GEN_STRUCTURED, GEN_RETRY, GEN_PARTIAL, GEN_FALLBACK}) == 4)

# A8 --------------------------------------------------------------------
print("\n--- A8: evidence support is measured ---")

supported = GameSearchEngine._evidence_support(
    [rec("1", evidence="relaxing farming simulation")],
    {"1": game("1", tags=["relaxing", "farming", "simulation"])},
)
check("citation matching the record scores 1.0", supported == 1.0, supported)

unsupported = GameSearchEngine._evidence_support(
    [rec("1", evidence="competitive esports shooter tournament")],
    {"1": game("1", tags=["relaxing", "farming"], desc="grow crops")},
)
check("unrelated citation scores 0.0", unsupported == 0.0, unsupported)

mixed = GameSearchEngine._evidence_support(
    [rec("1", evidence="relaxing farming"), rec("2", evidence="esports tournament")],
    {"1": game("1", tags=["relaxing", "farming"]), "2": game("2", tags=["puzzle"])},
)
check("mixed citations score in between", 0.0 < mixed < 1.0, mixed)
check("empty item list scores 0.0", GameSearchEngine._evidence_support([], {}) == 0.0)
check("empty citation is not counted as supported",
      GameSearchEngine._evidence_support(
          [rec("1", evidence="")], {"1": game("1")}) == 0.0)

# A6 --------------------------------------------------------------------
print("\n--- A6: telemetry contract ---")

for resp in [ok, hallucinated, short, None]:
    _, _, tele = engine_with([resp, resp]).generate_answer("q", matches())
    check(f"telemetry keys present ({str(resp)[:24]}...)",
          set(tele) == {"generation_mode", "rejected_app_ids", "retries",
                        "evidence_supported"}, tele)

print(f"\n{passed}/{passed + failed} passed")
raise SystemExit(1 if failed else 0)
