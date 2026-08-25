#!/usr/bin/env python3
"""Turn hand-written topical query specs into graded relevance judgements
(PRD-002, R4).

The hand-written part stays small -- a query, the tags that define what
answers it, and any hard constraints -- while the grading is derived from the
catalogue so it is reproducible and can be regenerated when the corpus changes.

Grades follow PRD-002 section 3.2:

    2  every required tag present
    1  at least one required tag and at least one nice-to-have tag
    0  otherwise

Two additions to that rule, introduced here because the five query categories
cannot be expressed without them:

* ``exclude_tags`` forces grade 0. Negation queries are one of the five
  categories, and without this a game tagged Cute would still count as a
  correct answer to "not aimed at children".
* ``filters`` force grade 0 when unmet. Hard-constraint queries ask about
  price, platform or multiplayer support, none of which are tags.

    uv run python eval/build_topical_queries.py
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "steam_games_reviews_25.sqlite"
SPEC_PATH = HERE / "topical_spec.jsonl"
OUT_PATH = HERE / "queries_topical.jsonl"

# Matches build_index.MIN_POSITIVE: judging games the index cannot return
# would put an unreachable ceiling on every configuration.
MIN_POSITIVE = 10


def load_catalogue(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT appid, name, price, release_date, windows, mac, linux,
               tags_json, categories_json
        FROM games
        WHERE name IS NOT NULL AND TRIM(name) != '' AND positive > ?
        """,
        (MIN_POSITIVE,),
    ).fetchall()

    catalogue = []
    for appid, name, price, release, win, mac, linux, tags_json, cats_json in rows:
        try:
            tags = json.loads(tags_json or "{}")
            tag_set = set(tags.keys() if isinstance(tags, dict) else tags)
        except Exception:
            tag_set = set()
        try:
            categories = set(json.loads(cats_json or "[]"))
        except Exception:
            categories = set()

        year = None
        if release:
            digits = [t for t in str(release).replace(",", " ").split() if t.isdigit() and len(t) == 4]
            if digits:
                year = int(digits[0])

        catalogue.append({
            "app_id": str(appid), "name": name, "price": price, "year": year,
            "windows": bool(win), "mac": bool(mac), "linux": bool(linux),
            "tags": tag_set, "categories": categories,
        })
    return catalogue


def passes_filters(game: dict, filters: dict) -> bool:
    if "price_max" in filters:
        price = game["price"]
        if price is None or price > filters["price_max"]:
            return False
    for platform in ("windows", "mac", "linux"):
        if filters.get(platform) and not game[platform]:
            return False
    if "released_after" in filters:
        if game["year"] is None or game["year"] < filters["released_after"]:
            return False
    if "released_before" in filters:
        if game["year"] is None or game["year"] >= filters["released_before"]:
            return False
    wanted = filters.get("categories_any")
    if wanted and not (set(wanted) & game["categories"]):
        return False
    return True


def grade(game: dict, spec: dict) -> int:
    if set(spec.get("exclude_tags", [])) & game["tags"]:
        return 0
    if not passes_filters(game, spec.get("filters", {})):
        return 0

    required = set(spec.get("required_tags", []))
    nice = set(spec.get("nice_tags", []))
    hit_required = required & game["tags"]

    if required and hit_required == required:
        return 2
    if hit_required and (nice & game["tags"]):
        return 1
    # Filter-only queries have no topical requirement; passing the filters is
    # the whole answer.
    if not required:
        return 2
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Grade topical queries from tag specs")
    ap.add_argument("--spec", type=Path, default=SPEC_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    specs = [json.loads(line) for line in args.spec.read_text().splitlines() if line.strip()]
    conn = sqlite3.connect(DB_PATH)
    catalogue = load_catalogue(conn)
    print(f"Catalogue: {len(catalogue):,} games with more than {MIN_POSITIVE} positive reviews")

    records, thin = [], []
    for spec in specs:
        grades = {}
        for game in catalogue:
            g = grade(game, spec)
            if g:
                grades[game["app_id"]] = g
        n2 = sum(1 for v in grades.values() if v == 2)
        if n2 < 5:
            thin.append((spec["query_id"], n2, len(grades)))
        records.append({
            "query_id": spec["query_id"],
            "query": spec["query"],
            "category": spec["category"],
            "track": "topical",
            "grades": grades,
        })

    args.out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
    )
    counts = [sum(1 for v in r["grades"].values() if v == 2) for r in records]
    print(f"Wrote {len(records)} queries to {args.out}")
    print(f"Grade-2 games per query: min {min(counts)}, median "
          f"{sorted(counts)[len(counts)//2]}, max {max(counts)}")
    if thin:
        print(f"\n{len(thin)} queries have fewer than 5 fully-relevant games:")
        for qid, n2, total in thin:
            print(f"  {qid}: {n2} at grade 2, {total} graded at all")


if __name__ == "__main__":
    main()
