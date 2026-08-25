#!/usr/bin/env python3
"""Join returned scores to the anonymisation key and summarise (PRD-005).

Reports mean and standard deviation across scoring passes per system, so a
difference between systems can be read against the variation in the judge
rather than taken at face value.

    uv run python eval/judge/aggregate_scores.py
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIMENSIONS = ("relevance", "verifiability", "satisfaction")
SYSTEM_ORDER = ("original", "revised", "chatgpt")
SYSTEM_LABEL = {
    "original": "Original system",
    "revised": "Revised system",
    "chatgpt": "ChatGPT",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", type=Path, default=HERE / "scoring_key.json")
    ap.add_argument("--scores", type=Path, nargs="*",
                    help="score files; defaults to scores_pass*.json in this directory")
    ap.add_argument("--out", type=Path, default=HERE / "judge_results.md")
    args = ap.parse_args()

    key = json.loads(args.key.read_text())
    files = args.scores or sorted(HERE.glob("scores_pass*.json"))
    if not files:
        raise SystemExit(
            "No score files found. Complete the passes described in "
            "scoring_packet.md and save them as scores_pass1.json, ..."
        )
    print(f"Scoring passes: {len(files)}  ({', '.join(f.name for f in files)})")

    # totals[system][round] = [total per pass]
    totals: dict[str, dict[str, list[float]]] = {}
    dims: dict[str, dict[str, list[float]]] = {}

    for path in files:
        data = json.loads(path.read_text())
        for rnd, by_label in data.items():
            mapping = key.get(str(rnd), {})
            for label, scores in by_label.items():
                system = mapping.get(label)
                if system is None:
                    print(f"  ! round {rnd} label {label} is not in the key; skipped")
                    continue
                total = sum(float(scores.get(d, 0)) for d in DIMENSIONS)
                totals.setdefault(system, {}).setdefault(str(rnd), []).append(total)
                for d in DIMENSIONS:
                    dims.setdefault(system, {}).setdefault(d, []).append(
                        float(scores.get(d, 0))
                    )

    def spread(values: list[float]) -> float:
        return st.stdev(values) if len(values) > 1 else 0.0

    rounds = sorted(key, key=int)
    lines = ["# Judge results", "",
             f"- Scoring passes: {len(files)}",
             "- Values are mean ± standard deviation across passes", ""]

    lines += ["| Round | " + " | ".join(SYSTEM_LABEL[s] for s in SYSTEM_ORDER if s in totals) + " |"]
    present = [s for s in SYSTEM_ORDER if s in totals]
    lines += ["|---" * (len(present) + 1) + "|"]
    for rnd in rounds:
        cells = []
        for s in present:
            vals = totals[s].get(rnd, [])
            cells.append(f"{st.mean(vals):.1f} ± {spread(vals):.1f}" if vals else "—")
        lines.append(f"| {rnd} | " + " | ".join(cells) + " |")

    means = []
    for s in present:
        per_round = [st.mean(v) for v in totals[s].values() if v]
        means.append(f"**{st.mean(per_round):.1f}**" if per_round else "—")
    lines.append("| **Mean** | " + " | ".join(means) + " |")

    lines += ["", "## By dimension", "",
              "| System | Relevance /30 | Verifiability /30 | Satisfaction /40 |",
              "|---|---|---|---|"]
    for s in present:
        cells = [
            f"{st.mean(dims[s][d]):.1f} ± {spread(dims[s][d]):.1f}"
            if dims.get(s, {}).get(d) else "—"
            for d in DIMENSIONS
        ]
        lines.append(f"| {SYSTEM_LABEL[s]} | " + " | ".join(cells) + " |")

    # Judge noise: how much a single pass can move a round's score.
    all_spreads = [spread(v) for s in present for v in totals[s].values() if len(v) > 1]
    if all_spreads:
        lines += ["", "## Measurement noise", "",
                  f"Median standard deviation across passes for a single round: "
                  f"**{st.median(all_spreads):.1f} points**. Differences between "
                  f"systems smaller than this should not be read as improvements."]

    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
