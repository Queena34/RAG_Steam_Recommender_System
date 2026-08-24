#!/usr/bin/env python3
"""Generate the known-item retrieval query set (PRD-002, R3).

Each query is a real player review; the correct answer is the game it was
written about. No annotation is needed and the label cannot be disputed.

Two properties make or break the validity of this set:

1. **The review must not already be in the index.** build_index.py embeds, per
   game, the top 15 positive and top 5 negative reviews within a length window,
   ranked by community votes. This script applies the identical ordering and
   takes only reviews *below* that cut. Getting this wrong would mean querying
   the index with text it already contains -- every score would rise and
   nothing would reveal why.

2. **The game's own name must be removed.** 30% of reviews name their game
   (measured over 300 samples). Left in, BM25 matches the string directly and
   the experiment measures string matching rather than retrieval.

    uv run python eval/build_knownitem_queries.py --n 500
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import build_index as B  # noqa: E402  (reuse the exact sampling constants)

DB_PATH = BASE_DIR / "steam_games_reviews_25.sqlite"
OUT_PATH = Path(__file__).resolve().parent / "queries_knownitem.jsonl"

# A query shorter than this after title removal carries too little signal to be
# a fair test of retrieval.
MIN_QUERY_CHARS = 80

PLACEHOLDER = "[title]"

# Words too generic to be worth stripping: removing them would gut the query
# without hiding the game's identity.
NAME_STOPWORDS = frozenset({
    "the", "of", "and", "a", "an", "in", "on", "for", "to", "with",
    "game", "edition", "remastered", "definitive", "deluxe", "complete",
    "collection", "ultimate", "goty", "hd", "vr", "ii", "iii", "iv", "vi",
    "vii", "viii", "ix", "xi", "xii",
})


def name_fragments(name: str) -> list[str]:
    """Distinctive fragments of a title that would give the answer away.

    Returns the full title first so it is replaced before its parts, then the
    individual content words, longest first -- otherwise a short word could be
    substituted inside a longer one that was still due for replacement.
    """
    cleaned = re.sub(r"[^\w\s]", " ", name)
    words = [w for w in cleaned.split() if len(w) >= 3 and w.lower() not in NAME_STOPWORDS]
    fragments = [name, cleaned.strip()] if cleaned.strip() != name else [name]
    fragments += sorted(set(words), key=len, reverse=True)
    return [f for f in fragments if f]


def strip_title(review: str, name: str) -> str:
    """Replace every mention of the game's title with a placeholder."""
    out = review
    for frag in name_fragments(name):
        out = re.sub(rf"\b{re.escape(frag)}\b", PLACEHOLDER, out, flags=re.IGNORECASE)
    # Collapse runs of placeholders and whitespace left behind.
    out = re.sub(rf"(?:{re.escape(PLACEHOLDER)}\s*)+", PLACEHOLDER + " ", out)
    return re.sub(r"\s+", " ", out).strip()


def eligible_games(conn: sqlite3.Connection, min_positive: int) -> list[int]:
    """Games that have at least one positive review beyond what is indexed."""
    rows = conn.execute(
        """
        SELECT r.appid
        FROM reviews r
        JOIN games g ON g.appid = r.appid
        WHERE g.positive > ?
          AND g.name IS NOT NULL AND TRIM(g.name) != ''
          AND r.voted_up = 1
          AND r.review IS NOT NULL
          AND LENGTH(TRIM(r.review)) BETWEEN ? AND ?
        GROUP BY r.appid
        HAVING COUNT(*) > ?
        """,
        (min_positive, B.REVIEW_MIN_CHARS, B.REVIEW_MAX_CHARS, B.MAX_POSITIVE_REVIEWS),
    ).fetchall()
    return [r[0] for r in rows]


def held_out_review(conn: sqlite3.Connection, appid: int) -> str | None:
    """The best positive review that build_index.py did *not* embed.

    Ordering is copied verbatim from fetch_reviews_for_batch(); the offset is
    MAX_POSITIVE_REVIEWS, so this is rank 16 -- the highest-voted review that
    falls outside the indexed window.
    """
    row = conn.execute(
        """
        SELECT review FROM reviews
        WHERE appid = ?
          AND voted_up = 1
          AND review IS NOT NULL
          AND LENGTH(TRIM(review)) BETWEEN ? AND ?
        ORDER BY votes_up DESC, LENGTH(review) DESC
        LIMIT 1 OFFSET ?
        """,
        (appid, B.REVIEW_MIN_CHARS, B.REVIEW_MAX_CHARS, B.MAX_POSITIVE_REVIEWS),
    ).fetchone()
    return row[0] if row else None


def main() -> None:
    ap = argparse.ArgumentParser(description="Build known-item evaluation queries")
    ap.add_argument("--n", type=int, default=500, help="number of queries")
    ap.add_argument("--seed", type=int, default=20260824, help="sampling seed")
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    conn = sqlite3.connect(DB_PATH)

    print(f"Index sampling rules in force: positive rank <= {B.MAX_POSITIVE_REVIEWS}, "
          f"length {B.REVIEW_MIN_CHARS}-{B.REVIEW_MAX_CHARS}, min_positive {B.MIN_POSITIVE}")

    candidates = eligible_games(conn, B.MIN_POSITIVE)
    print(f"Games with a held-out positive review: {len(candidates):,}")

    rng = random.Random(args.seed)
    rng.shuffle(candidates)

    written = 0
    too_short = 0
    records = []
    for appid in candidates:
        if written >= args.n:
            break
        review = held_out_review(conn, appid)
        if not review:
            continue
        name = conn.execute("SELECT name FROM games WHERE appid = ?", (appid,)).fetchone()[0]
        query = strip_title(review, name)
        if len(query) < MIN_QUERY_CHARS:
            too_short += 1
            continue
        records.append({
            "query_id": f"ki-{appid}",
            "query": query,
            "gold_appid": str(appid),
            "gold_name": name,
            "track": "knownitem",
        })
        written += 1

    args.out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
    )
    print(f"Wrote {written} queries to {args.out}")
    print(f"Skipped {too_short} reviews that were too short after title removal")


if __name__ == "__main__":
    main()
