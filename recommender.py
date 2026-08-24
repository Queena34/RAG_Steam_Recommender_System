from __future__ import annotations

import json
import math
import os
import random
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("RAGLOOKER_DB_PATH", BASE_DIR / "steam_games_reviews_25.sqlite"))
INDEX_DIR = BASE_DIR / "vector_index"

DEFAULT_MATCH_COUNT = 5
RETRIEVAL_COUNT = 50        # candidates kept after fusion, before re-ranking
RETRIEVAL_POOL = 100        # candidates each retriever contributes to the fusion
RRF_K = 60                  # Reciprocal Rank Fusion smoothing constant

# Blend of "how well does it match the query" vs "is it a good game".
RELEVANCE_WEIGHT = 0.75
QUALITY_WEIGHT = 0.25

# Review-aware quality scoring
WILSON_Z = 1.96             # 95% confidence for the Wilson lower bound
POPULARITY_REF = 500_000    # review count treated as "maximally popular"
RATING_WEIGHT = 0.6         # rating vs popularity split inside the quality score

# Diversity constraints applied to the final ranking
MAX_PER_DEVELOPER = 2
MAX_PER_TAG = 3
MAX_FALLBACK_GAMES = 5000   # games loaded for SQLite-LIKE keyword-search fallback
BM25_MAX_GAMES = 0          # 0 = all games; set to N to limit BM25 corpus size
LLM_TIMEOUT = 150           # seconds before giving up on an LLM call


def create_search_engine() -> "GameSearchEngine":
    return GameSearchEngine(DB_PATH)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class GameRecord:
    app_id: str
    raw: dict[str, Any]

    @property
    def name(self) -> str:
        return self.raw.get("name", "Unknown title")

    @property
    def short_description(self) -> str:
        return self.raw.get("short_description", "")

    def to_result(self, score: float) -> dict[str, Any]:
        return {
            "app_id": self.app_id,
            "name": self.name,
            "score": round(score, 4),
            "short_description": self.short_description,
            "genres": self.raw.get("genres", []),
            "tags": self._normalize_tags(self.raw.get("tags")),
            "price": self.raw.get("price"),
            "release_date": self.raw.get("release_date"),
            "header_image": self.raw.get("header_image"),
            "store_page": f"https://store.steampowered.com/app/{self.app_id}",
            "platforms": {
                "windows": bool(self.raw.get("windows")),
                "mac": bool(self.raw.get("mac")),
                "linux": bool(self.raw.get("linux")),
            },
        }

    @staticmethod
    def _normalize_tags(tags: Any) -> list[str]:
        if isinstance(tags, dict):
            return list(tags.keys())[:8]
        if isinstance(tags, list):
            return tags[:8]
        return []


# ---------------------------------------------------------------------------
# Search engine
# ---------------------------------------------------------------------------

class GameSearchEngine:
    """
    RAG-based Steam game recommendation engine.

    Retrieval: FAISS vector search (Ollama nomic-embed-text embeddings)
               → SQLite keyword search fallback if index unavailable
    Ranking:   Ollama LLM (phi4-mini / gemma3 / …)
               → similarity-score fallback
    Answer:    Ollama LLM natural-language response
               → simple formatted text fallback
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

        # Components initialised below; None means "not available"
        self.faiss_index = None          # faiss.Index
        self.faiss_app_ids: list[str] = []
        self.embed_fn = None             # (str) → list[float]
        self.embed_model: str | None = None
        self.query_prefix: str = ""      # set from meta.json by _init_vector_index
        self.llm_model: str | None = None

        # BM25 index (built at startup from all games in SQLite)
        self._bm25 = None
        self._bm25_app_ids: list[str] = []

        # Fallback: small in-memory game list for keyword-LIKE search
        self.records: list[GameRecord] = []

        self._init_vector_index()
        self._init_embed_fn()
        self._init_llm()
        self._init_bm25()

        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            print("FAISS index unavailable – using BM25 retrieval.")
        else:
            print(
                f"Ready: {self.faiss_index.ntotal} games FAISS-indexed, "
                f"embed={self.embed_model}, llm={self.llm_model or 'none'}"
            )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_vector_index(self) -> None:
        index_path = INDEX_DIR / "index.faiss"
        meta_path = INDEX_DIR / "meta.json"
        if not index_path.exists():
            print("No FAISS index found. Run build_index.py first.")
            return
        try:
            import faiss
            self.faiss_index = faiss.read_index(str(index_path))
            meta = json.loads(meta_path.read_text())
            self.faiss_app_ids = meta["app_ids"]
            self._index_meta = meta  # keep for embed-model cross-check

            # FAISS returns row numbers; app_ids[i] is the only thing that maps
            # row i back to a game. A length mismatch means every lookup past
            # the divergence point resolves to the wrong game, with no error
            # anywhere -- so treat the index as unusable instead of loading it.
            if self.faiss_index.ntotal != len(self.faiss_app_ids):
                raise ValueError(
                    f"index/metadata mismatch: {self.faiss_index.ntotal} vectors "
                    f"but {len(self.faiss_app_ids)} app_ids"
                )

            # nomic-embed-text is asymmetric: documents and queries must carry
            # different task prefixes. Read the prefix back from the index
            # metadata rather than hard-coding it, so an index built without
            # prefixes (which reports none) keeps being queried without them.
            # Prefixing only one side is worse than prefixing neither.
            self.query_prefix = meta.get("query_prefix", "")
            print(f"FAISS index loaded: {self.faiss_index.ntotal} vectors")
            if self.query_prefix:
                print(f"Query prefix: {self.query_prefix!r}")
        except Exception as e:
            # Reset partial state: faiss_index is assigned before the metadata
            # is validated, so leaving it set here would keep a half-loaded or
            # mismatched index in play and defeat the check above.
            self.faiss_index = None
            self.faiss_app_ids = []
            self.query_prefix = ""
            print(f"FAISS load failed: {e}")

    def _init_embed_fn(self) -> None:
        """Use the same embedding model that was used to build the index."""
        meta = getattr(self, "_index_meta", {})
        source = meta.get("source", "")

        if source == "sentence-transformers":
            try:
                from sentence_transformers import SentenceTransformer
                st_model = SentenceTransformer("all-MiniLM-L6-v2")

                def embed_st(text: str) -> list[float]:
                    return st_model.encode([text])[0].tolist()

                self.embed_fn = embed_st
                self.embed_model = "sentence-transformers:all-MiniLM-L6-v2"
                print("Embed model: sentence-transformers all-MiniLM-L6-v2")
                return
            except Exception as e:
                print(f"sentence-transformers not available: {e}")
                return

        # Fallback: Ollama (for indexes built with nomic-embed-text)
        required_model = meta.get("model", "nomic-embed-text")
        required_dim = meta.get("dimension")
        try:
            import ollama
            test = ollama.embed(model=required_model, input="test")
            actual_dim = len(test.embeddings[0])

            if required_dim and actual_dim != required_dim:
                print(
                    f"Dimension mismatch: index has {required_dim}-d, "
                    f"model produces {actual_dim}-d. Vector search disabled."
                )
                return

            model = required_model

            def embed_ollama(text: str) -> list[float]:
                return ollama.embed(model=model, input=text).embeddings[0]

            self.embed_fn = embed_ollama
            self.embed_model = f"ollama:{model}"
        except Exception as e:
            print(f"Ollama embedding not available ({e}). Vector search disabled.")

    def _init_llm(self) -> None:
        """Detect the best available Ollama chat/generate model."""
        try:
            import ollama
            available = [m.model for m in ollama.list().models]
            preference = [
                "phi4-mini", "gemma3:4b", "gemma3:2b", "gemma3",
                "phi3.5", "phi3", "llama3.2:3b", "llama3.2",
                "mistral", "qwen2.5:3b", "qwen2.5",
            ]
            for want in preference:
                for avail in available:
                    if want in avail:
                        self.llm_model = avail
                        print(f"LLM model: {avail}")
                        return
            if available:
                self.llm_model = available[0]
                print(f"LLM model: {available[0]} (first available)")
        except Exception as e:
            print(f"Ollama LLM not available: {e}")

    def _init_bm25(self) -> None:
        """Build a BM25 index from all games in SQLite (runs at startup)."""
        try:
            from rank_bm25 import BM25Okapi
            import nltk
            nltk.download("punkt", quiet=True)
            try:
                from nltk.stem import PorterStemmer
                stemmer = PorterStemmer()
                def tokenize(text: str) -> list[str]:
                    tokens = re.split(r"\W+", text.lower())
                    return [stemmer.stem(t) for t in tokens if len(t) > 1]
            except ImportError:
                def tokenize(text: str) -> list[str]:
                    return [t for t in re.split(r"\W+", text.lower()) if len(t) > 1]
            self._bm25_tokenize = tokenize

            cache_path = BASE_DIR / "bm25_cache.pkl"
            if cache_path.exists():
                try:
                    import pickle
                    with open(cache_path, "rb") as f:
                        self._bm25, self._bm25_app_ids = pickle.load(f)
                    print(f"BM25 index loaded from cache: {len(self._bm25_app_ids)} games")
                    return
                except Exception as e:
                    print(f"BM25 cache load failed ({e}), rebuilding...")

            limit_clause = f"LIMIT {BM25_MAX_GAMES}" if BM25_MAX_GAMES else ""
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    f"""
                    SELECT appid, name, short_description, genres_json, tags_json
                    FROM games
                    WHERE name IS NOT NULL AND TRIM(name) != ''
                    ORDER BY appid
                    {limit_clause}
                    """
                ).fetchall()

            app_ids: list[str] = []
            corpus: list[list[str]] = []
            for row in rows:
                text_parts = [row["name"] or ""]

                # Repeat the short description twice to increase its weight
                if row["short_description"]:
                    text_parts.append(row["short_description"])
                    text_parts.append(row["short_description"])

                # Parse tags and repeat them twice, since tags best represent the game type
                try:
                    tags = json.loads(row["tags_json"] or "{}")
                    tag_list = list(tags.keys()) if isinstance(tags, dict) else tags
                    tag_text = " ".join(tag_list[:20])
                    text_parts.append(tag_text)
                    text_parts.append(tag_text)
                except Exception:
                    pass

                # Parse genres
                try:
                    genres = json.loads(row["genres_json"] or "[]")
                    text_parts.append(" ".join(genres))
                except Exception:
                    pass

                tokens = tokenize(" ".join(text_parts))
                app_ids.append(str(row["appid"]))
                corpus.append(tokens)

            self._bm25 = BM25Okapi(corpus)
            self._bm25_app_ids = app_ids
            print(f"BM25 index built: {len(app_ids)} games")

            try:
                import pickle
                with open(cache_path, "wb") as f:
                    pickle.dump((self._bm25, self._bm25_app_ids), f)
                print(f"BM25 cache saved to {cache_path}")
            except Exception as e:
                print(f"BM25 cache save failed: {e}")
        except Exception as e:
            print(f"BM25 init failed: {e}")

    def _load_fallback_records(self) -> list[GameRecord]:
        from steam_sqlite import load_games_from_sqlite
        return [
            GameRecord(app_id=app_id, raw=raw)
            for app_id, raw in load_games_from_sqlite(self.db_path, MAX_FALLBACK_GAMES)
        ]

    # ------------------------------------------------------------------
    # Public API (shape must stay stable)
    # ------------------------------------------------------------------

    def search(self, query: str) -> dict[str, Any]:
        candidates = self.retrieve_candidates(query)
        ranked_matches = self.rank_candidates(query, candidates)

        has_vector = bool(self.faiss_index and self.faiss_index.ntotal > 0 and self.embed_fn)
        has_bm25 = bool(self._bm25)
        if has_vector and has_bm25:
            mode = "hybrid-rrf"
        elif has_vector:
            mode = "vector"
        elif has_bm25:
            mode = "bm25"
        else:
            mode = "keyword-fallback"
        if mode != "keyword-fallback" and self.llm_model:
            mode += "+llm-answer"

        answer, selected_names = self.generate_answer(query, ranked_matches)

        if selected_names:
            # Reorder ranked_matches to match LLM selection order.
            # Use startswith fuzzy match since LLM may truncate long names.
            name_to_match = {rec.name: (rec, score) for rec, score in ranked_matches}
            final_matches: list[tuple[GameRecord, float]] = []
            for sel in selected_names:
                for name, pair in name_to_match.items():
                    if name.startswith(sel) or sel.startswith(name):
                        final_matches.append(pair)
                        break
            if not final_matches:
                final_matches = ranked_matches[:5]
        else:
            final_matches = ranked_matches[:5]

        if len(final_matches) < 5:
            # LLM did not return enough games — fall back entirely to score-based
            # ranking and simple answer to keep cards and text consistent
            final_matches = ranked_matches[:5]
            answer = self._simple_answer(query, final_matches)

        results = [record.to_result(score) for record, score in final_matches]

        return {
            "matches": results,
            "answer": answer,
            "meta": {
                "indexed_games": (
                    self.faiss_index.ntotal if self.faiss_index else len(self.records)
                ),
                "retrieval_mode": mode,
                "embed_model": self.embed_model or "none",
                "llm_model": self.llm_model or "none",
            },
        }

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _expand_query_with_llm(self, query: str) -> str:
        if not self.llm_model:
            return query
        try:
            import ollama
            resp = ollama.generate(
                model=self.llm_model,
                prompt=(
                    f'Given this Steam game search query: "{query}"\n'
                    f'List 8-10 related gaming keywords, synonyms, and tags '
                    f'that would appear in Steam game descriptions, genres, or tags.\n'
                    f'IMPORTANT: if the query contains negative constraints such as '
                    f'"without combat", "less farming", "no multiplayer", or "not X", '
                    f'do NOT include the excluded concepts in your keywords.\n'
                    f'For example: if query is "relaxing farming", '
                    f'keywords might be: cozy, harvest, crops, agriculture, '
                    f'casual, peaceful, simulation, garden, animals, life-sim\n'
                    f'Reply ONLY with comma-separated keywords. No explanation.'
                ),
                options={"temperature": 0.3, "num_predict": 60},
            )
            expanded_terms = resp.response.strip()
            full_query = f"{query} {expanded_terms}"
            print(f"Query expanded: {full_query}")
            return full_query
        except Exception as e:
            print(f"Query expansion failed: {e}, using original query")
            return query

    def retrieve_candidates(self, query: str) -> list[GameRecord]:
        """Hybrid retrieval: FAISS (semantic) + BM25 (lexical), fused with RRF.

        Both retrievers run on every query and each returns a ranked list of
        app_ids. The lists are merged with Reciprocal Rank Fusion, which
        combines *ranks* rather than raw scores.

        This replaces the previous fallback chain, where BM25 was only reached
        if FAISS was unavailable -- meaning the lexical index was built at
        startup but never used in practice. Fusing on rank also removes the
        need to compare FAISS cosine similarity against min-max-normalised
        BM25 scores, which live on incompatible scales.

        If only one retriever is available its ranking is used unchanged; if
        neither is, we fall back to the SQLite LIKE search.
        """
        expanded_query = self._expand_query_with_llm(query)

        rank_lists: list[list[str]] = []

        if self.faiss_index and self.embed_fn and self.faiss_index.ntotal > 0:
            try:
                rank_lists.append(self._retrieve_vector_ids(expanded_query, RETRIEVAL_POOL))
            except Exception as e:
                print(f"Vector search error: {e}")

        if self._bm25:
            try:
                rank_lists.append(self._retrieve_bm25_ids(expanded_query, RETRIEVAL_POOL))
            except Exception as e:
                print(f"BM25 search error: {e}")

        rank_lists = [lst for lst in rank_lists if lst]
        if not rank_lists:
            return self._retrieve_keyword(expanded_query)

        fused = self._rrf_fuse(rank_lists, RRF_K)[:RETRIEVAL_COUNT]
        app_ids = [app_id for app_id, _ in fused]
        fused_scores = dict(fused)

        records_map = self._fetch_game_details(app_ids)

        ordered: list[GameRecord] = []
        for app_id in app_ids:
            rec = records_map.get(app_id)
            if rec is not None:
                rec.raw["_rrf"] = fused_scores[app_id]
                ordered.append(rec)

        self._assign_retrieval_scores(ordered)
        return sorted(ordered, key=lambda r: r.raw["_score"], reverse=True)

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf_fuse(rank_lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
        """Reciprocal Rank Fusion over several ranked app_id lists.

        Each list contributes 1 / (k + rank) to every app_id it contains
        (rank is 1-based). Because only ranks are used, retrievers with
        wildly different score distributions can be combined without any
        normalisation. k dampens the influence of the very top positions;
        k=60 is the value from the original RRF paper (Cormack et al., 2009).

        Returns (app_id, score) pairs sorted by descending fused score.
        Ties are broken by app_id so the ordering is deterministic.
        """
        scores: dict[str, float] = {}
        for lst in rank_lists:
            for rank, app_id in enumerate(lst, 1):
                scores[app_id] = scores.get(app_id, 0.0) + 1.0 / (k + rank)
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))

    @staticmethod
    def _assign_retrieval_scores(records: list[GameRecord]) -> None:
        """Blend fused relevance with review-aware quality into `_score`.

        The RRF score is normalised against the best candidate in this result
        set, so `_score` stays comparable across queries.
        """
        if not records:
            return
        max_rrf = max(r.raw.get("_rrf", 0.0) for r in records) or 1.0
        for rec in records:
            relevance = rec.raw.get("_rrf", 0.0) / max_rrf
            quality = GameSearchEngine._quality_score(rec)
            rec.raw["_relevance"] = relevance
            rec.raw["_quality"] = quality
            rec.raw["_score"] = RELEVANCE_WEIGHT * relevance + QUALITY_WEIGHT * quality

    @staticmethod
    def _quality_score(record: GameRecord) -> float:
        """Review-aware quality signal in [0, 1].

        Combines two things the raw positive-review count conflates:

        * how well reviewed the game is -- as the Wilson lower bound of the
          positive ratio, not the raw ratio. A game with 3/3 positive reviews
          scores far below one with 47,000/50,000, which is what we want: the
          raw ratio would rank the former higher on almost no evidence.
        * how popular it is -- log review volume, so the difference between
          100 and 1,000 reviews matters more than 100k vs 500k.

        Games with no reviews get 0.0 rather than a neutral score, so they
        never outrank a game with real evidence behind it at equal relevance.
        """
        pos = record.raw.get("positive") or 0
        neg = record.raw.get("negative") or 0
        total = pos + neg
        if total <= 0:
            return 0.0

        # Wilson score lower bound for a Bernoulli parameter.
        phat = pos / total
        z = WILSON_Z
        denom = 1 + z * z / total
        centre = phat + z * z / (2 * total)
        margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
        wilson = max(0.0, (centre - margin) / denom)

        popularity = min(1.0, math.log1p(total) / math.log1p(POPULARITY_REF))

        return RATING_WEIGHT * wilson + (1 - RATING_WEIGHT) * popularity

    # ------------------------------------------------------------------
    # Individual retrievers (return ranked app_ids, no DB round-trip)
    # ------------------------------------------------------------------

    def _retrieve_vector_ids(self, query: str, k: int) -> list[str]:
        import faiss

        q_vec = np.array([self.embed_fn(self.query_prefix + query)], dtype=np.float32)
        faiss.normalize_L2(q_vec)

        n = min(k, self.faiss_index.ntotal)
        _scores, indices = self.faiss_index.search(q_vec, n)
        return [self.faiss_app_ids[i] for i in indices[0] if i >= 0]

    def _retrieve_bm25_ids(self, query: str, k: int) -> list[str]:
        tokenize = getattr(
            self,
            "_bm25_tokenize",
            lambda t: [x for x in re.split(r"\W+", t.lower()) if len(x) > 1],
        )
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        return [self._bm25_app_ids[i] for i in top_indices if scores[i] > 0]

    # Common words that add no search value
    _STOP_WORDS = frozenset({
        "the", "and", "for", "with", "that", "this", "are", "was",
        "has", "have", "its", "you", "your", "from", "but", "not",
        "can", "all", "good", "great", "best", "some", "more",
    })

    def _retrieve_keyword(self, query: str) -> list[GameRecord]:
        """SQLite LIKE search: AND-first (precise), OR fallback (broad)."""
        raw_words = [w.lower() for w in re.split(r"\W+", query) if len(w) > 2]
        keywords = [w for w in raw_words if w not in self._STOP_WORDS][:6]
        if not keywords:
            keywords = raw_words[:4]
        if not keywords:
            return random.sample(self.records, min(DEFAULT_MATCH_COUNT, len(self.records)))

        def _corpus_col():
            return (
                "LOWER(name || ' ' || COALESCE(short_description,'') "
                "|| ' ' || COALESCE(tags_json,'') "
                "|| ' ' || COALESCE(genres_json,''))"
            )

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                base_select = """
                    SELECT appid, name, short_description, release_date, price,
                           header_image, windows, mac, linux,
                           developers_json, publishers_json, categories_json,
                           genres_json, tags_json, user_score, positive, negative
                    FROM games WHERE name IS NOT NULL
                """
                # Try AND first (all keywords must appear)
                and_cond = " AND ".join(f"{_corpus_col()} LIKE ?" for _ in keywords)
                params = [f"%{kw}%" for kw in keywords]
                rows = conn.execute(
                    f"{base_select} AND ({and_cond}) LIMIT {RETRIEVAL_COUNT}",
                    params,
                ).fetchall()

                # Fallback: OR (any keyword matches)
                if len(rows) < DEFAULT_MATCH_COUNT:
                    or_cond = " OR ".join(f"{_corpus_col()} LIKE ?" for _ in keywords)
                    rows = conn.execute(
                        f"{base_select} AND ({or_cond}) LIMIT {RETRIEVAL_COUNT}",
                        params,
                    ).fetchall()

            if rows:
                return [self._row_to_record(row) for row in rows]
        except Exception as e:
            print(f"Keyword search error: {e}")

        if self.records:
            return random.sample(self.records, min(DEFAULT_MATCH_COUNT, len(self.records)))
        return []

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank_candidates(
        self, query: str, candidates: list[GameRecord]
    ) -> list[tuple[GameRecord, float]]:
        """Re-rank fused candidates: hard filters, preference bonus, diversity.

        Input scores come from retrieve_candidates(), where RRF relevance is
        already blended with the review-aware quality signal. On top of that:

        1. Hard filters drop candidates violating explicit platform / free
           constraints.
        2. A small preference bonus rewards soft constraints (cheap, recent,
           retro) that lexical and semantic retrieval treat as weak signals.
        3. A diversity pass caps how many results may share a developer or a
           primary tag, so the shortlist is not five near-identical games.

        Generative LLM re-ranking is still not used here: on CPU it costs
        ~550s per query, dominated by prompt prefill over 50 candidates.
        _rank_with_llm() below retains that implementation for GPU hosts.
        """
        if not candidates:
            return []

        # [ADDED] Parse explicit user preferences from the query string once,
        # then apply a small score bonus to each candidate that satisfies them.
        preferences = GameSearchEngine._extract_preferences(query)

        # Hard filters: remove candidates that don't satisfy explicit platform
        # or free-to-play constraints. These are non-negotiable requirements,
        # unlike price/date preferences which remain as soft bonuses.
        filtered = candidates
        if preferences["free"]:
            free_candidates = [r for r in candidates if r.raw.get("price") == 0]
            if free_candidates:  # only apply filter if it doesn't empty the list
                filtered = free_candidates
        for platform in ("windows", "mac", "linux"):
            if preferences[platform]:
                platform_candidates = [r for r in filtered if r.raw.get(platform)]
                if platform_candidates:  # only apply filter if it doesn't empty the list
                    filtered = platform_candidates

        ranked = sorted(
            [
                (r, r.raw.get("_score", 0.0) + GameSearchEngine._preference_bonus(preferences, r))
                for r in filtered
            ],
            key=lambda x: x[1],
            reverse=True,
        )
        return GameSearchEngine._apply_diversity(ranked, limit=10)

    @staticmethod
    def _apply_diversity(
        ranked: list[tuple[GameRecord, float]],
        limit: int,
        max_per_developer: int = MAX_PER_DEVELOPER,
        max_per_tag: int = MAX_PER_TAG,
    ) -> list[tuple[GameRecord, float]]:
        """Greedily pick `limit` results, capping developer and tag repeats.

        Relevance order is preserved: we walk the ranked list once and take
        each candidate unless it would exceed a cap, in which case it is held
        back. If the caps leave us short of `limit`, the held-back candidates
        are appended in their original order, so this can only reorder
        results -- never return fewer than the unconstrained ranking would.

        Without this, a query like "farming sim" tends to return five games
        from the same studio or the same sub-genre, which is technically
        relevant but useless as a recommendation set.
        """
        selected: list[tuple[GameRecord, float]] = []
        deferred: list[tuple[GameRecord, float]] = []
        dev_counts: dict[str, int] = {}
        tag_counts: dict[str, int] = {}

        for rec, score in ranked:
            if len(selected) >= limit:
                deferred.append((rec, score))
                continue

            developers = rec.raw.get("developers") or []
            dev = str(developers[0]).lower() if developers else ""
            tags = rec._normalize_tags(rec.raw.get("tags"))
            tag = str(tags[0]).lower() if tags else ""

            if dev and dev_counts.get(dev, 0) >= max_per_developer:
                deferred.append((rec, score))
                continue
            if tag and tag_counts.get(tag, 0) >= max_per_tag:
                deferred.append((rec, score))
                continue

            selected.append((rec, score))
            if dev:
                dev_counts[dev] = dev_counts.get(dev, 0) + 1
            if tag:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        if len(selected) < limit:
            selected.extend(deferred[: limit - len(selected)])
        return selected

    # ------------------------------------------------------------------
    # [ADDED] Preference bonus — ported and adapted from codex_modified_backup/recommender.py
    # Both methods were originally module-level functions; moved into the class as @staticmethod for consistency.
    #
    # WHY this exists:
    #   Semantic (FAISS) and lexical (BM25) retrieval treat query constraints like
    #   "free", "cheap", "linux", or "retro" as soft signals mixed in with many
    #   other terms. A game that is literally free-to-play may score lower than a
    #   paid game that just uses the word "free" in its description. These two
    #   small functions fix that by reading explicit constraints directly from the
    #   query string and rewarding games that actually satisfy them.
    #
    # HOW it works:
    #   1. _extract_preferences(query) → dict[str, bool]
    #      Does a simple substring scan on the lowercased query and returns a
    #      boolean flag for each constraint category:
    #        - free      : query contains "free" or "no cost"
    #        - cheap     : query contains "cheap", "budget", "low price", "inexpensive"
    #        - windows   : query contains "windows" or "pc"
    #        - mac       : query contains "mac"
    #        - linux     : query contains "linux" or "steam deck"
    #        - recent    : query contains "new", "recent", or "modern"
    #        - classic   : query contains "old", "classic", or "retro"
    #
    #   2. _preference_bonus(preferences, record) → float
    #      For each active flag, adds a fixed bonus to the candidate's score
    #      if the game's metadata satisfies the constraint:
    #        - free game (price == 0)              → +0.12
    #        - cheap game (price ≤ $10)            → +0.08
    #        - supports the requested platform     → +0.05 per platform
    #        - released ≥ 2020 when "recent"       → +0.05
    #        - released ≤ 2015 when "classic"      → +0.05
    #
    # PRACTICAL IMPACT:
    #   For queries like "free horror game", "cheap indie on steam deck", or
    #   "classic retro RPG", the bonus nudges games that literally satisfy the
    #   constraint to the top over games that merely mention the word in passing.
    #   For queries with no explicit constraints all flags are False and the bonus
    #   is 0.0, so ordinary searches are completely unaffected.
    # ------------------------------------------------------------------

    @staticmethod
    # All single-word checks use \b word-boundary regex (re.escape) to prevent false positives;
    # multi-word phrases ("no cost", "low price", "steam deck") use plain `in` checks.
    def _extract_preferences(query: str) -> dict:
        lowered = query.lower()

        def wb(word: str) -> bool:
            return bool(re.search(r'\b' + re.escape(word) + r'\b', lowered))

        return {
            "free":    (wb("free") and not wb("roam")) or "no cost" in lowered,  # excludes "free roam" / "free movement"
            "cheap":   wb("cheap") or wb("budget") or wb("inexpensive") or "low price" in lowered,
            "windows": wb("windows") or wb("pc"),
            "mac":     wb("mac"),
            "linux":   wb("linux") or "steam deck" in lowered,
            "recent":  wb("new") or wb("recent") or wb("modern"),
            "classic": wb("old") or wb("classic") or wb("retro"),
        }

    @staticmethod
    def _preference_bonus(preferences: dict, record: GameRecord) -> float:
        """
        [ADDED] Return a small additive score bonus for a candidate game that
        satisfies one or more explicit constraints detected in the query.
        Returns 0.0 when no preference flags are active (most queries).
        """
        bonus = 0.0

        price = record.raw.get("price")

        # Budget / cheap: anything at or under $10
        if preferences["cheap"] and isinstance(price, (int, float)) and price <= 10:
            bonus += 0.08

        # Release-year constraints: parse year from the release_date string
        release_date = str(record.raw.get("release_date") or "")
        year_match = re.search(r"(19|20)\d{2}", release_date)
        if year_match:
            year = int(year_match.group(0))
            if preferences["recent"] and year >= 2020:
                bonus += 0.05
            if preferences["classic"] and year <= 2015:
                bonus += 0.05

        # Cap at 0.15 so stacked bonuses never override semantic relevance scores.
        return min(bonus, 0.15)

    @staticmethod
    def _call_with_timeout(fn, timeout_sec: float, *args, **kwargs):
        """Run fn(*args, **kwargs) in a thread; raise TimeoutError if too slow."""
        result: list[Any] = [None]
        exc: list[BaseException | None] = [None]

        def _target():
            try:
                result[0] = fn(*args, **kwargs)
            except Exception as e:
                exc[0] = e

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join(timeout=timeout_sec)
        if t.is_alive():
            raise TimeoutError(f"LLM call exceeded {timeout_sec}s")
        if exc[0]:
            raise exc[0]
        return result[0]

    # _rank_with_llm() is fully implemented but disabled at runtime.
    # On CPU hardware, the 50-candidate prompt prefill takes approximately
    # 550 seconds per query, making it impractical for interactive use.
    # With GPU acceleration this approach would be viable.
    # Final candidate selection has been moved to generate_answer(), where
    # the LLM selects the best 5 from 10 ranked candidates and generates
    # natural language explanations simultaneously.
    def _rank_with_llm(
        self, query: str, candidates: list[GameRecord]
    ) -> list[tuple[GameRecord, float]]:
        import ollama

        prompt = (
            f'A user is looking for Steam games matching: "{query}"\n\n'
            f"Here are {len(candidates)} candidate games:\n\n"
        )
        for i, r in enumerate(candidates, 1):
            tags = ", ".join(r._normalize_tags(r.raw.get("tags"))[:8])
            genres = ", ".join(r.raw.get("genres", []))
            pos = r.raw.get("positive") or 0
            neg = r.raw.get("negative") or 0
            total = pos + neg
            rating = f"{round(100*pos/total)}% positive ({total} reviews)" if total > 0 else "unknown rating"
            desc = r.short_description[:200]
            reviews = r.raw.get("review_summary", "")[:300]
            prompt += f"{i}. [{r.app_id}] {r.name}\n"
            prompt += f"   Genres: {genres} | Tags: {tags}\n"
            prompt += f"   Rating: {rating}\n"
            prompt += f"   Description: {desc}\n"
            if reviews:
                prompt += f"   Player reviews: {reviews}\n"
            prompt += "\n"

        prompt += (
            f'Select the {DEFAULT_MATCH_COUNT} games that BEST match "{query}". '
            f"Consider gameplay mechanics, genre fit, mood, and player feedback.\n"
            f'Reply ONLY with JSON array: [{{"rank":1,"app_id":"ID","reason":"2 sentences why this matches"}},...]\n'
            f"No explanation outside the JSON."
        )

        resp = ollama.generate(
            model=self.llm_model,
            prompt=prompt,
            options={"temperature": 0.1, "num_predict": 800},
        )
        return self._parse_llm_ranking(resp.response.strip(), candidates)

    def _parse_llm_ranking(
        self, text: str, candidates: list[GameRecord]
    ) -> list[tuple[GameRecord, float]]:
        by_id = {r.app_id: r for r in candidates}

        # Try JSON extraction
        try:
            m = re.search(r"\[[\s\S]*?\]", text)
            if m:
                items = json.loads(m.group())
                result: list[tuple[GameRecord, float]] = []
                for item in items[:DEFAULT_MATCH_COUNT]:
                    aid = str(item.get("app_id", "")).strip()
                    reason = item.get("reason", "")
                    rank = item.get("rank", len(result) + 1)
                    if aid in by_id:
                        rec = by_id[aid]
                        rec.raw["_llm_reason"] = reason
                        score = max(0.5, 1.0 - (rank - 1) * 0.08)
                        result.append((rec, score))
                if result:
                    return result
        except Exception:
            pass

        # Name-matching fallback
        result = []
        for rec in candidates:
            if rec.name.lower() in text.lower():
                result.append((rec, rec.raw.get("_score", 0.5)))
        if result:
            return sorted(result, key=lambda x: x[1], reverse=True)[:DEFAULT_MATCH_COUNT]

        # Score-based fallback
        return sorted(
            [(r, r.raw.get("_score", 0.5)) for r in candidates[:DEFAULT_MATCH_COUNT]],
            key=lambda x: x[1],
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Answer generation
    # ------------------------------------------------------------------

    def generate_answer(self, query: str, matches: list[tuple[GameRecord, float]]) -> tuple[str, list[str]]:
        if not matches:
            return "No games found matching your description.", []

        if self.llm_model:
            try:
                return self._call_with_timeout(
                    self._generate_with_llm, LLM_TIMEOUT, query, matches
                )
            except Exception as e:
                print(f"LLM answer failed/timed out: {e}")

        return self._simple_answer(query, matches), []

    def _generate_with_llm(
        self, query: str, matches: list[tuple[GameRecord, float]]
    ) -> str:
        import ollama

        # Keep prompt short: fewer input tokens = faster prefill on CPU
        lines = []
        for i, (rec, _) in enumerate(matches, 1):
            tags = ", ".join(rec._normalize_tags(rec.raw.get("tags"))[:4])
            desc = rec.short_description[:100]
            lines.append(f"{i}. {rec.name} [{tags}] — {desc}")

        prompt = (
            'IMPORTANT: You MUST only recommend games from the candidate list '
            'provided below. Do NOT recommend any games from your own knowledge '
            'that are not in this list. Do NOT invent or hallucinate game titles. '
            'Only use the exact game names as they appear in the list below.\n\n'
            f'A user is looking for Steam games matching: "{query}"\n\n'
            f'Here are 10 candidate games:\n'
            + "\n".join(lines)
            + '\n\nYou MUST select exactly 5 games, no more and no less. '
            'Even if some candidates are not a perfect fit, still include them '
            'and briefly explain what partial value they offer. '
            'Never recommend fewer than 5 games under any circumstances. '
            'For each selected game, write 1-2 sentences explaining specifically '
            'why it fits the request. Enthusiastic tone, plain text, no bullet points, '
            'present them in order from best to worst match. '
            'Format each recommendation exactly as: '
            '1. Game Name [tags] — explanation sentence. '
            'Use this numbered format strictly.'
        )

        resp = ollama.generate(
            model=self.llm_model,
            prompt=prompt,
            options={"temperature": 0.7, "num_predict": 800},
        )
        response = resp.response.strip()
        patterns = [
            r'^\d+\.\s+([^[\n\-–]+)',   # "1. Game Name [" or "1. Game Name -"
            r'^([^[\n\-–]+)\s*[\-–]',   # "Game Name -" or "Game Name –"
            r'^([^[\n]+)\[',             # "Game Name ["
        ]
        selected_names = []
        for pattern in patterns:
            matches = re.findall(pattern, response, re.MULTILINE)
            if matches:
                selected_names = [m.strip() for m in matches[:5]]
                break
        return response, selected_names

    def _simple_answer(self, query: str, matches: list[tuple[GameRecord, float]]) -> str:
        names = ", ".join(f'"{r.name}"' for r, _ in matches[:3])
        return (
            f'For "{query}", the closest matches are: {names}. '
            "Results are ranked by semantic similarity to your description."
        )

    # ------------------------------------------------------------------
    # SQLite helpers
    # ------------------------------------------------------------------

    def _fetch_game_details(self, app_ids: list[str]) -> dict[str, GameRecord]:
        if not app_ids:
            return {}
        int_ids = [int(a) for a in app_ids]
        ph = ",".join("?" * len(int_ids))

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            game_rows = conn.execute(
                f"""
                SELECT appid, name, short_description, release_date, price, header_image,
                       windows, mac, linux, developers_json, publishers_json,
                       categories_json, genres_json, tags_json, user_score, positive, negative
                FROM games WHERE appid IN ({ph})
                """,
                int_ids,
            ).fetchall()

            review_rows = conn.execute(
                f"""
                SELECT appid, review FROM (
                    SELECT appid, review,
                           ROW_NUMBER() OVER (
                               PARTITION BY appid ORDER BY LENGTH(review) DESC
                           ) AS rn
                    FROM reviews
                    WHERE appid IN ({ph})
                      AND voted_up = 1
                      AND review IS NOT NULL
                      AND LENGTH(TRIM(review)) > 30
                ) WHERE rn <= 8
                """,
                int_ids,
            ).fetchall()

        reviews_map: dict[str, list[str]] = {}
        for row in review_rows:
            aid = str(row["appid"])
            reviews_map.setdefault(aid, []).append(row["review"][:200])

        result: dict[str, GameRecord] = {}
        for row in game_rows:
            app_id = str(row["appid"])
            summary = " | ".join(reviews_map.get(app_id, []))
            result[app_id] = self._row_to_record(row, summary)

        return result

    def _row_to_record(self, row: sqlite3.Row, review_summary: str = "") -> GameRecord:
        def jload(v, default):
            if not v:
                return default
            try:
                return json.loads(v)
            except Exception:
                return default

        raw: dict[str, Any] = {
            "name": row["name"] or "Unknown",
            "short_description": row["short_description"] or "",
            "release_date": row["release_date"],
            "price": row["price"],
            "header_image": row["header_image"],
            "windows": bool(row["windows"]),
            "mac": bool(row["mac"]),
            "linux": bool(row["linux"]),
            "developers": jload(row["developers_json"], []),
            "publishers": jload(row["publishers_json"], []),
            "categories": jload(row["categories_json"], []),
            "genres": jload(row["genres_json"], []),
            "tags": jload(row["tags_json"], {}),
            "user_score": row["user_score"],
            "positive": row["positive"],
            "negative": row["negative"],
            "review_summary": review_summary,
        }
        return GameRecord(app_id=str(row["appid"]), raw=raw)
