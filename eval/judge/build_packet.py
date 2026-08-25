#!/usr/bin/env python3
"""Assemble the blind scoring packet for the judge protocol.

Three systems are compared per round: the revised system, the original system
as recorded in report Table 14, and ChatGPT as recorded there too. Labels are
shuffled independently per round, so a judge cannot infer the source from
position, and the key is written to a separate file that is not handed over.

Re-scoring the two recorded systems is not optional. Their published figures
came from a single scoring pass; comparing a fresh multi-pass mean against them
would mix a change in the system with a change in the measurement.

    uv run python eval/judge/build_packet.py --passes 3
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent

RUBRIC = """Score each recommendation set out of 100, as the sum of three dimensions:

  Relevance to the prompt                                        30 points
  Verifiability on the Steam store                               30 points
  Projected player satisfaction, from an experienced gamer's
  perspective                                                    40 points

Judge only what is shown. Do not attempt to identify which system produced
which set. Give the three sub-scores for every set, plus one sentence of
justification."""


# Preference order when a round has several runs. A run that fell back to
# ranking order is not the behaviour under test -- packaging one would mean
# scoring the fallback path and reporting it as the revised system.
MODE_PREFERENCE = ["structured", "structured-retry", "structured-partial", "fallback-ranking"]


def load_system_outputs(paths: list[Path]) -> tuple[dict[int, list[str]], dict[int, str]]:
    """Best available run per round, with the mode it came from.

    Later files override earlier ones for the same round, so a re-run can be
    passed after the original capture.
    """
    best: dict[int, tuple[int, list[str], str]] = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            titles = rec.get("titles")
            if not titles:
                continue
            mode = (rec.get("meta") or {}).get("generation_mode", "fallback-ranking")
            rank = MODE_PREFERENCE.index(mode) if mode in MODE_PREFERENCE else len(MODE_PREFERENCE)
            rnd = rec["round"]
            if rnd not in best or rank < best[rnd][0]:
                best[rnd] = (rank, titles, mode)
    return ({r: v[1] for r, v in best.items()},
            {r: v[2] for r, v in best.items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=3,
                    help="independent scoring passes the packet asks for")
    ap.add_argument("--seed", type=int, default=20260825)
    ap.add_argument("--system", type=Path, nargs="+",
                    default=[HERE / "system_outputs.jsonl",
                             HERE / "system_outputs_r5.jsonl"])
    ap.add_argument("--baseline", type=Path, default=HERE / "baseline_outputs.json")
    ap.add_argument("--packet", type=Path, default=HERE / "scoring_packet.md")
    ap.add_argument("--key", type=Path, default=HERE / "scoring_key.json")
    args = ap.parse_args()

    revised, modes = load_system_outputs(list(args.system))
    baseline = json.loads(args.baseline.read_text())
    rng = random.Random(args.seed)

    lines = [
        "# Blind scoring packet",
        "",
        f"Five rounds, three anonymous recommendation sets each. Please score the "
        f"whole packet **{args.passes} times**, in **separate conversations**, "
        "without referring back to earlier passes.",
        "",
        "The sets come from different recommender systems. Which is which is not "
        "disclosed, and the ordering changes between rounds. Please judge only "
        "what each set contains.",
        "",
        "## Why several passes",
        "",
        "An LLM judge is not deterministic. A single pass cannot separate a real "
        "difference between systems from variation in the scoring itself. With "
        f"{args.passes} passes the spread across passes can be reported next to "
        "the means, so a claimed improvement can be checked against the noise in "
        "the measurement.",
        "",
        "## Rubric",
        "",
        "```",
        RUBRIC,
        "```",
        "",
        "## Rounds",
        "",
    ]

    key = {}
    for entry in baseline["rounds"]:
        rnd = entry["round"]
        sets = [
            ("revised", revised.get(rnd, [])),
            ("original", entry["original_system"]["titles"]),
            ("chatgpt", entry["chatgpt"]["titles"]),
        ]
        sets = [s for s in sets if s[1]]
        rng.shuffle(sets)

        labels = ["A", "B", "C"][: len(sets)]
        key[str(rnd)] = {label: source for label, (source, _) in zip(labels, sets)}

        lines += [f"### Round {rnd} — {entry['category']}", "",
                  f"**Prompt:** {entry['query']}", ""]
        for label, (_, titles) in zip(labels, sets):
            lines.append(f"**Set {label}**")
            lines += [f"- {t}" for t in titles]
            lines.append("")
        lines.append("")

    lines += [
        "---",
        "",
        "## How to answer",
        "",
        "Reply with **one JSON object and nothing else** — no commentary before "
        "or after it. Every round and every set must appear, with all three "
        "sub-scores as numbers.",
        "",
        "```json",
        "{",
        '  "1": {',
        '    "A": {"relevance": 0, "verifiability": 0, "satisfaction": 0, "note": "one sentence"},',
        '    "B": {"relevance": 0, "verifiability": 0, "satisfaction": 0, "note": "one sentence"},',
        '    "C": {"relevance": 0, "verifiability": 0, "satisfaction": 0, "note": "one sentence"}',
        "  },",
        '  "2": { "A": {...}, "B": {...}, "C": {...} }',
        "}",
        "```",
        "",
        "Totals are computed from the sub-scores, so do not include them.",
        "Save the reply verbatim as `scores_pass1.json` (then `2`, `3`) in "
        "`eval/judge/`, and run `aggregate_scores.py`.",
    ]

    args.packet.write_text("\n".join(lines) + "\n")
    args.key.write_text(json.dumps(key, indent=2) + "\n")

    print(f"Packet written to {args.packet}")
    print(f"Key written to {args.key}  (do not include this in what the judge sees)")
    print("\nRevised-system run selected per round:")
    for rnd in sorted(modes):
        flag = "" if modes[rnd] == "structured" else "   <-- not a clean run"
        print(f"  round {rnd}: {modes[rnd]}{flag}")

    missing = [e["round"] for e in baseline["rounds"] if not revised.get(e["round"])]
    if missing:
        print(f"\nWARNING: no revised-system output for round(s) {missing}; "
              f"those rounds compare only two sets.")


if __name__ == "__main__":
    main()
