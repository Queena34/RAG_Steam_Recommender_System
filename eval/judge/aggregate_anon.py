#!/usr/bin/env python3
"""Compare named against attribute-only scoring (E2).

Tests whether the judge scores partly by recognising titles. The same three
recommendation sets were scored twice: once with titles, once with titles
replaced by attributes. If recognition drives the gap between systems, hiding
names should move well-known sets down and obscure ones up.

    uv run python eval/judge/aggregate_anon.py
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIMS = ("relevance", "verifiability", "satisfaction")
ORDER = ("original", "revised", "chatgpt")
LABEL = {"original": "Original system", "revised": "Revised system", "chatgpt": "ChatGPT"}


def collect(key_path: Path, pattern: str) -> dict[str, dict[str, list[float]]]:
    key = json.loads(key_path.read_text())
    files = sorted(HERE.glob(pattern))
    if not files:
        raise SystemExit(f"No score files matching {pattern}")
    out: dict[str, dict[str, list[float]]] = {}
    for path in files:
        data = json.loads(path.read_text())
        for rnd, by_label in data.items():
            for lab, scores in by_label.items():
                system = key.get(str(rnd), {}).get(lab)
                if system is None:
                    continue
                total = sum(float(scores.get(d, 0)) for d in DIMS)
                out.setdefault(system, {}).setdefault(str(rnd), []).append(total)
    return out, len(files)


def mean_by_round(data):
    return {s: {r: st.mean(v) for r, v in rounds.items()} for s, rounds in data.items()}


def main() -> None:
    named, n_named = collect(HERE / "round2_w060" / "scoring_key.json", "round2_w060/scores_pass*.json")
    anon, n_anon = collect(HERE / "anon_key.json", "anon_pass*.json")
    nm, am = mean_by_round(named), mean_by_round(anon)
    rounds = sorted(set().union(*[set(v) for v in nm.values()]), key=int)

    lines = ["# Named against attribute-only scoring", "",
             f"- Named passes: {n_named}   attribute-only passes: {n_anon}",
             "- See `anon_limitations.md` before reading these numbers", "",
             "| System | Named | Attributes only | Change |", "|---|---|---|---|"]
    deltas = {}
    for s in ORDER:
        if s not in nm or s not in am:
            continue
        a = st.mean(nm[s].values()) if hasattr(nm[s], "values") else 0
        a = st.mean(list(nm[s].values()))
        b = st.mean(list(am[s].values()))
        deltas[s] = b - a
        lines.append(f"| {LABEL[s]} | {a:.1f} | {b:.1f} | **{b - a:+.1f}** |")

    lines += ["", "## Per round", "",
              "| Round | " + " | ".join(f"{LABEL[s]}" for s in ORDER if s in nm) + " |",
              "|---" * (len([s for s in ORDER if s in nm]) + 1) + "|"]
    for r in rounds:
        cells = []
        for s in ORDER:
            if s not in nm:
                continue
            a, b = nm[s].get(r), am[s].get(r)
            cells.append(f"{a:.0f} → {b:.0f} ({b - a:+.0f})" if a is not None and b is not None else "—")
        lines.append(f"| {r} | " + " | ".join(cells) + " |")

    # Round 5 carries a partial anonymisation leak; check the effect without it.
    if "5" in rounds:
        lines += ["", "## Excluding round 5", "",
                  "Round 5's prompt names a game that also appears as an entry "
                  "(see `anon_limitations.md`).", "",
                  "| System | Named | Attributes only | Change |", "|---|---|---|---|"]
        for s in ORDER:
            if s not in nm:
                continue
            rs = [r for r in rounds if r != "5" and r in nm[s] and r in am[s]]
            a = st.mean([nm[s][r] for r in rs])
            b = st.mean([am[s][r] for r in rs])
            lines.append(f"| {LABEL[s]} | {a:.1f} | {b:.1f} | **{b - a:+.1f}** |")

    if "chatgpt" in deltas and "revised" in deltas:
        spread = deltas["chatgpt"] - deltas["revised"]
        lines += ["", "## Reading", "",
                  f"ChatGPT moves {deltas['chatgpt']:+.1f} and the revised system "
                  f"{deltas['revised']:+.1f} when titles are hidden, a relative shift of "
                  f"{spread:+.1f} points.", "",
                  "A negative relative shift means the better-known set lost more "
                  "from anonymisation than the obscure one, which is what a "
                  "recognition effect predicts. A shift near zero means the gap "
                  "rests on the attributes rather than on familiarity, and the "
                  "argument in OPTIMIZED_DESIGN §11.3 does not hold here.", "",
                  "Judge drift measured on fixed content between scoring batches "
                  "is 2.5 points (PRD-005 §11), so shifts smaller than that are "
                  "not distinguishable from it."]

    out = HERE / "anon_results.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
