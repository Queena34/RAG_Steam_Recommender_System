"""Regression tests for P0 reliability fixes.

    uv run python test_reliability.py
"""
from __future__ import annotations

import sqlite3
import tempfile
from types import SimpleNamespace
from pathlib import Path

import app as app_module
from recommender import GameSearchEngine, GameRecord


passed = failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"✅ {name}")
    else:
        failed += 1
        print(f"❌ {name}  {detail}")


def make_db() -> Path:
    handle = tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False)
    handle.close()
    path = Path(handle.name)
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE games (
            appid INTEGER PRIMARY KEY, name TEXT, short_description TEXT,
            release_date TEXT, price REAL, header_image TEXT,
            windows INTEGER, mac INTEGER, linux INTEGER,
            developers_json TEXT, publishers_json TEXT, categories_json TEXT,
            genres_json TEXT, tags_json TEXT, user_score INTEGER,
            positive INTEGER, negative INTEGER
        )"""
    )
    conn.executemany(
        "INSERT INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (3, "Puzzle Farm", "relaxing farming puzzle", "2020", 5, "", 1, 0, 0,
            "[]", "[]", '["Single-player"]', '["Simulation"]', '{"Farming Sim": 1}', 0, 100, 10),
            (2, "Farm Horror", "farming horror survival", "2021", 5, "", 1, 0, 0,
            "[]", "[]", '["Multi-player"]', '["Horror"]', '{"Farming Sim": 1}', 0, 90, 10),
        ],
    )
    conn.commit()
    conn.close()
    return path


db = make_db()
try:
    engine = object.__new__(GameSearchEngine)
    engine.db_path = db
    engine.records = []
    intent = engine._rule_query_intent("free Linux single-player game without multiplayer")
    check("rules extract price, platform and mode constraints", (
        intent["price_max"] == 0.0
        and intent["platforms"] == ["linux"]
        and intent["modes"] == ["single-player"]
        and intent["exclude_modes"] == ["multiplayer"]
    ))
    allowed = engine._hard_filter_ids(intent)
    check("hard filters produce an exact catalogue allow-list", allowed == set())
    single_intent = engine._rule_query_intent("single-player farming game")
    check("single-player filter reaches the database", engine._hard_filter_ids(single_intent) == {"3"})

    class FakeTokenizer:
        def encode(self, query, document):
            n = 3 if "Puzzle" in document else 2
            return SimpleNamespace(ids=list(range(n)), attention_mask=[1] * n, type_ids=[0] * n)

    class FakeSession:
        def get_inputs(self):
            return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")]

        def run(self, _, inputs):
            return [[2.0, 1.0]]

    engine.reranker_tokenizer = FakeTokenizer()
    engine.reranker_session = FakeSession()
    reranked = engine._rerank_cross_encoder(
        "farming game", [GameRecord("2", {"name": "Farm Horror"}), GameRecord("3", {"name": "Puzzle Farm"})]
    )
    check("cross-encoder returns one score per candidate", reranked is not None and len(reranked) == 2)
    check("cross-encoder scores are normalized", reranked is not None and 0 < reranked[0][1] < 1)

    engine.reranker_session = None
    check("missing cross-encoder degrades cleanly", engine._rerank_cross_encoder("q", []) is None)

    result = engine._retrieve_keyword("farming horror")
    check("keyword fallback prioritizes the all-term match", [r.app_id for r in result] == ["2", "3"])
    check("keyword fallback assigns ranking scores", all("_score" in r.raw for r in result))
    check("keyword fallback scores are finite", all(r.raw["_score"] >= 0 for r in result))

    # The fallback must not expose arbitrary exception text through the API.
    original_factory = app_module.create_search_engine

    class BrokenEngine:
        def search(self, query):
            raise RuntimeError("/private/path/secret.sqlite")

    app_module.create_search_engine = lambda: BrokenEngine()
    client = app_module.create_app().test_client()
    bad_shape = client.post("/api/search", json={"query": 123})
    check("non-string query is rejected", bad_shape.status_code == 400)
    failure = client.post("/api/search", json={"query": "test"})
    check("search failure returns generic response", failure.status_code == 500)
    check("search failure hides exception details", "secret.sqlite" not in failure.get_data(as_text=True))
    app_module.create_search_engine = original_factory
finally:
    db.unlink(missing_ok=True)

print(f"\n{passed}/{passed + failed} passed")
raise SystemExit(1 if failed else 0)
