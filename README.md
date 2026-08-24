# Advanced Analytics G28 — Project 3: RAG-Based Steam Game Recommender

A retrieval-augmented generation (RAG) system that recommends Steam games based on natural language queries, combining hybrid vector + keyword retrieval with LLM-based reranking and answer generation.

---

## Dataset

The project data lives in the `steam_games_reviews_25.sqlite` SQLite database (not tracked in git due to size).

Data about games on the Steam shop was scraped up until the beginning of April 2026. Only games with more than 25 reviews were kept. For each game, only the most recent 500 English reviews were scraped.

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
│  LLM Reranking + Hard Filters   │  ← phi4-mini via Ollama
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

**Fallback chain:** if FAISS index is unavailable → BM25 only; if LLM is unavailable → similarity-score ranking + formatted text output.

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
| `rank_candidates()` | LLM-based reranking with hard filters (price, platform, genre) |
| `generate_answer()` | LLM generates a natural language recommendation with reasoning |
| `search()` | Orchestrates the full pipeline; returns a fixed JSON shape consumed by the Flask API |

### `build_index.py`

Embeds all game descriptions using `nomic-embed-text` (via Ollama) and stores the resulting vectors in a FAISS index. This step is run once and may take several hours depending on the number of games.

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

The SQLite database is not included in this repository due to its size. Place `steam_games_reviews_25.sqlite` in the project root directory.

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
