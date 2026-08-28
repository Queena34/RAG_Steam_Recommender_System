# Advanced Analytics G28 — Project 3: RAG-Based Steam Game Recommender

A retrieval-augmented generation (RAG) system that recommends Steam games from natural-language queries. It combines FAISS semantic retrieval, BM25 lexical retrieval, deterministic score-based ranking, and Ollama-based structured recommendation generation.

---

## Dataset

The project data lives in the `steam_games_reviews_25.sqlite` SQLite database (not tracked in git due to size).

Data about games on the Steam shop was scraped up until the beginning of April 2026. The database contains 39,176 game records and 7,679,845 reviews. The current dense index includes games with `positive > 10` (30,693 games); BM25 is built over the full catalogue.

| Table | Description |
|-------|-------------|
| `games` | Steam game metadata (name, description, genres, tags, price, platforms, etc.) |
| `reviews` | Steam user reviews used for retrieval and sentiment-aware recommendations |

---

## System Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────┐
│  Hybrid Retrieval (Top 50)      │
│  ├── FAISS vector search        │  ← nomic-embed-text embeddings
│  └── BM25 keyword search        │  ← pickle-cached at startup
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  RRF + quality ranking          │  ← relevance, Wilson quality, preferences, diversity
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  LLM Answer Generation          │  ← natural language recommendation
└─────────────────────────────────┘
    │
    ▼
Flask API → Web Frontend
```

Both FAISS and BM25 run when available. Their top-100 ranked lists are fused with RRF, then reduced to 50 candidates. If one retriever is unavailable, the other is used; if neither is available, SQLite `LIKE` search is used. LLM generation has a structured-output retry and a plain-text fallback.

An optional ONNX Cross-encoder can rerank the 50 fused candidates before the final score-based ranking. Set `RAGLOOKER_RERANKER_MODEL_DIR` to a directory containing a pre-exported `model.onnx` and `tokenizer.json`; if the model is missing or fails, the API reports the status and uses the deterministic ranking path.

---

## Repository Structure

```
├── recommender.py          # Core RAG engine (GameSearchEngine)
├── build_index.py          # Embeds all games and builds FAISS index
├── app.py                  # Flask API server
├── steam_sqlite.py         # SQLite database interface
├── test_preferences.py     # Tests for preference extraction
├── templates/index.html    # Web frontend
├── static/
│   ├── app.js              # Frontend logic
│   └── styles.css          # Styling
├── vector_index/
│   ├── index.faiss         # Pre-built FAISS vector index
│   └── meta.json           # Index metadata (model, dimension, app_ids)
├── pyproject.toml          # Python dependencies (managed by uv)
└── uv.lock
```

---

## Key Components

### `recommender.py` — `GameSearchEngine`

| Method | Description |
|--------|-------------|
| `retrieve_candidates()` | Hybrid FAISS + BM25 retrieval; returns top 50 candidates |
| `rank_candidates()` | Applies explicit free/platform filters when possible, score-based preference bonuses, review-aware quality ranking, and diversity constraints |
| `generate_answer()` | LLM selects candidate `app_id`s and produces reasons/evidence through a JSON schema; falls back when unavailable |
| `search()` | Orchestrates the full pipeline; returns a fixed JSON shape consumed by the Flask API |

### `build_index.py`

Builds one rich document per eligible game using `nomic-embed-text` via Ollama. Documents include metadata, gameplay modes, rating information, and sampled positive/negative reviews. The resulting 768-dimensional vectors are stored in a FAISS `IndexFlatIP` index. This step is run once and may take several hours.

---

## Installation and Running

### 1. Install dependencies

Make sure the package manager `uv` is installed ([https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)), then run:

```bash
uv sync
```

### 2. Set up Ollama

Make sure [Ollama](https://ollama.com/) is installed and running, then pull the required models:

```bash
ollama pull nomic-embed-text
ollama pull phi4-mini
```

### 3. Add the database file

The SQLite database is not included in this repository due to its size. Place `steam_games_reviews_25.sqlite` in the project root directory, or set `RAGLOOKER_DB_PATH` to its location.

### 4. Build the FAISS index

This step embeds all game data into a vector index and may take several hours:

```bash
uv run python build_index.py
```

### 5. Start the Flask app

```bash
uv run flask --app app run --debug
```

Then open `http://127.0.0.1:5000`.
