#!/usr/bin/env python3
"""Build the title-anonymised scoring packet (E2).

Tests one claim: that the judge scores partly by recognising titles rather than
by what the games are. The same three recommendation sets are presented again,
with every title replaced by a code and all objective attributes kept -- tags,
genres, modes, rating, review count, price, release year, description.

Review counts are deliberately retained. The claim under test is about
*recognition*, not popularity, and popularity is legitimate evidence for
projected satisfaction. Keeping it makes the test conservative: if scores still
converge once names are hidden, recognition was doing the work.

Descriptions are scrubbed of the title, since a description naming its own game
would defeat the anonymisation.

    uv run python eval/judge/build_anon_packet.py
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

DB_PATH = BASE_DIR / "steam_games_reviews_25.sqlite"

# Titles the snapshot stores under a different string, or a predecessor entry
# that is clearly the same product. Resolved explicitly rather than by fuzzy
# matching, so the mapping is auditable.
NAME_OVERRIDES = {
    "Helldivers 2": "HELLDIVERS™ 2",
    "Life is Feudal": "Life is Feudal: Your Own",
}

GAMEPLAY_CATEGORIES = {
    "Single-player", "Multi-player", "Co-op", "Online Co-op", "LAN Co-op",
    "Shared/Split Screen Co-op", "PvP", "Online PvP", "LAN PvP",
    "Shared/Split Screen PvP", "MMO", "Cross-Platform Multiplayer",
}

RUBRIC = """Score each recommendation set out of 100, as the sum of three dimensions:

  Relevance to the prompt                                        30 points
  Verifiability on the Steam store                               30 points
  Projected player satisfaction, from an experienced gamer's
  perspective                                                    40 points

Titles are withheld. Judge each entry on the attributes given."""


def scrub(text: str, name: str) -> str:
    """Remove the title and its distinctive words from a description."""
    out = text
    words = [w for w in re.sub(r"[^\w\s]", " ", name).split() if len(w) >= 4]
    for frag in [name] + sorted(set(words), key=len, reverse=True):
        out = re.sub(rf"\b{re.escape(frag)}\b", "the game", out, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", out).strip()


def attributes(conn: sqlite3.Connection, name: str) -> dict | None:
    lookup = NAME_OVERRIDES.get(name, name)
    row = conn.execute(
        """SELECT appid, name, short_description, price, release_date,
                  positive, negative, tags_json, genres_json, categories_json
           FROM games WHERE name = ?""",
        (lookup,),
    ).fetchone()
    if row is None:
        return None

    appid, real_name, desc, price, release, pos, neg, tags_j, genres_j, cats_j = row
    try:
        tags = json.loads(tags_j or "{}")
        tag_list = list(tags.keys() if isinstance(tags, dict) else tags)[:8]
    except Exception:
        tag_list = []
    try:
        genres = json.loads(genres_j or "[]")[:4]
    except Exception:
        genres = []
    try:
        modes = [c for c in json.loads(cats_j or "[]") if c in GAMEPLAY_CATEGORIES]
    except Exception:
        modes = []

    total = (pos or 0) + (neg or 0)
    return {
        "tags": tag_list,
        "genres": genres,
        "modes": modes,
        "rating": f"{round(100 * pos / total)}% positive ({total:,} reviews)" if total else "no rating data",
        "price": "free" if price == 0 else (f"{price:.2f} EUR" if price else "unknown"),
        "release": release or "unknown",
        "description": scrub((desc or "")[:220], real_name),
    }


def render(code: str, attrs: dict | None) -> list[str]:
    if attrs is None:
        return [f"**{code}** — not present in the catalogue snapshot; "
                f"attributes unavailable", ""]
    lines = [f"**{code}**"]
    if attrs["tags"]:
        lines.append(f"- Tags: {', '.join(attrs['tags'])}")
    if attrs["genres"]:
        lines.append(f"- Genres: {', '.join(attrs['genres'])}")
    if attrs["modes"]:
        lines.append(f"- Modes: {', '.join(attrs['modes'])}")
    lines.append(f"- Rating: {attrs['rating']}")
    lines.append(f"- Price: {attrs['price']}   Released: {attrs['release']}")
    if attrs["description"]:
        lines.append(f"- Description: {attrs['description']}")
    lines.append("")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--system", type=Path, default=HERE / "system_outputs_w060.jsonl")
    ap.add_argument("--baseline", type=Path, default=HERE / "baseline_outputs.json")
    ap.add_argument("--packet", type=Path, default=HERE / "anon_packet.md")
    ap.add_argument("--key", type=Path, default=HERE / "anon_key.json")
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)
    baseline = json.loads(args.baseline.read_text())

    pref = ["structured", "structured-retry", "structured-partial", "fallback-ranking"]
    best: dict[int, list[str]] = {}
    rank_of: dict[int, int] = {}
    for line in args.system.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if not rec.get("titles"):
            continue
        mode = (rec.get("meta") or {}).get("generation_mode", "fallback-ranking")
        r = pref.index(mode) if mode in pref else len(pref)
        if rec["round"] not in rank_of or r < rank_of[rec["round"]]:
            rank_of[rec["round"]] = r
            best[rec["round"]] = rec["titles"]

    rng = random.Random(args.seed)
    lines = [
        "# Blind scoring packet — attributes only",
        "",
        f"Five rounds, three anonymous recommendation sets each. Please score the "
        f"whole packet **{args.passes} times**, in **separate conversations**, "
        "without referring back to earlier passes.",
        "",
        "Game titles are withheld. Each entry is given by its store attributes. "
        "Judge only what is shown, and do not try to work out which game an "
        "entry is.",
        "",
        "## Rubric",
        "",
        "```", RUBRIC, "```",
        "",
        "## Rounds",
        "",
    ]

    key: dict[str, dict] = {}
    unavailable = []
    for entry in baseline["rounds"]:
        rnd = entry["round"]
        sets = [
            ("revised", best.get(rnd, [])),
            ("original", entry["original_system"]["titles"]),
            ("chatgpt", entry["chatgpt"]["titles"]),
        ]
        sets = [s for s in sets if s[1]]
        rng.shuffle(sets)
        labels = ["A", "B", "C"][: len(sets)]
        key[str(rnd)] = {lab: src for lab, (src, _) in zip(labels, sets)}

        lines += [f"### Round {rnd} — {entry['category']}", "",
                  f"**Prompt:** {entry['query']}", ""]
        for lab, (src, titles) in zip(labels, sets):
            lines += [f"#### Set {lab}", ""]
            for i, title in enumerate(titles, 1):
                attrs = attributes(conn, title)
                if attrs is None:
                    unavailable.append((rnd, src, title))
                lines += render(f"{lab}{i}", attrs)
        lines.append("")

    lines += [
        "---", "",
        "## How to answer", "",
        "Reply with **one JSON object and nothing else**, scoring each set:", "",
        "```json",
        "{",
        '  "1": {',
        '    "A": {"relevance": 0, "verifiability": 0, "satisfaction": 0, "note": "one sentence"},',
        '    "B": {...}, "C": {...}',
        "  },",
        '  "2": { ... }',
        "}",
        "```", "",
        "Save each pass as `anon_pass1.json`, `anon_pass2.json`, `anon_pass3.json`.",
    ]

    args.packet.write_text("\n".join(lines) + "\n")
    args.key.write_text(json.dumps(key, indent=2) + "\n")
    print(f"Packet written to {args.packet}")
    print(f"Key written to {args.key}  (withhold from the judge)")
    if unavailable:
        print(f"\n{len(unavailable)} entries have no attributes in the snapshot:")
        for rnd, src, title in unavailable:
            print(f"  round {rnd}  {src}: {title}")
        print("  These are shown as unavailable rather than invented.")


if __name__ == "__main__":
    main()
