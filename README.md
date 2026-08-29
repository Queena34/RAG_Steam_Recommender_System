# Steam Game Recommender

A retrieval-augmented Steam game recommendation system that turns natural-language preferences into explainable game suggestions.

Users can describe the experience they want—such as “a relaxing single-player farming game”—and receive a ranked list of Steam games with a concise explanation of why each result fits.

## Highlights

- Natural-language game discovery instead of keyword-only search
- Hybrid retrieval with FAISS semantic search and BM25 lexical search
- Structured query understanding for platform, price, and multiplayer constraints
- Cross-encoder reranking when an ONNX model is configured
- Review-aware quality scoring and result diversification
- Grounded, user-facing explanations for every recommendation
- Graceful fallback to local retrieval and deterministic explanations when optional services are unavailable

## Architecture

```text
Natural-language query
        │
        ▼
Structured query understanding
        │
        ▼
FAISS + BM25 hybrid retrieval
        │
        ▼
Hard filters + quality-aware ranking
        │
        ▼
Optional Cross-encoder reranking
        │
        ▼
Structured recommendation explanations
        │
        ▼
Flask API + responsive web interface
```

The retrieval and ranking path remains usable without an LLM. Optional LLM providers add structured query parsing and natural-language explanations; deterministic fallbacks keep the application predictable when a provider is unavailable.

## Technology

| Area | Technology |
|---|---|
| Web application | Flask, HTML, CSS, JavaScript |
| Semantic retrieval | FAISS, Ollama `nomic-embed-text` |
| Lexical retrieval | BM25 |
| Reranking | Optional ONNX Cross-encoder |
| LLM integration | Ollama or DeepSeek OpenAI-compatible API |
| Data storage | SQLite |
| Dependency management | uv |

## Quick start

### Requirements

- Python 3.13+
- [uv](https://docs.astral.sh/uv/)
- A Steam catalogue SQLite database named `steam_games_reviews_25.sqlite`, or a path supplied through `RAGLOOKER_DB_PATH`
- Ollama with `nomic-embed-text` for semantic retrieval

Install dependencies:

```bash
uv sync
```

Copy the configuration template if you want to customize local paths or providers:

```bash
cp .env.example .env
```

Place the catalogue database in the project root, then build the semantic index:

```bash
uv run python build_index.py
```

Start the application:

```bash
uv run flask --app app run --debug
```

Open <http://127.0.0.1:5000> in a browser.

### Optional language-model configuration

For a fast local smoke test, disable chat generation:

```bash
RAGLOOKER_LLM_ENABLED=0 uv run flask --app app run --debug
```

To use DeepSeek for structured parsing and explanations:

```bash
RAGLOOKER_LLM_PROVIDER=deepseek \
DEEPSEEK_API_KEY=your-api-key \
uv run flask --app app run --debug
```

The API key should only be supplied through the environment and must never be committed.

## API

```http
POST /api/search
Content-Type: application/json

{"query": "a relaxing single-player farming game"}
```

Each result includes game metadata, a normalized ranking score, `why_recommended`, and an optional `caveat` for trade-offs such as multiplayer support in a single-player request. The legacy `answer` field remains in the response for compatibility.

## Evaluation and tests

The repository includes focused regression tests for query preferences, ranking, structured generation, reliability, and offline retrieval metrics:

```bash
uv run python test_preferences.py
uv run python test_ranking.py
uv run python test_generation.py
uv run python test_reliability.py
uv run python eval/test_metrics.py
```

See [`docs/architecture.md`](docs/architecture.md) for the public implementation overview and [`docs/evaluation.md`](docs/evaluation.md) for the evaluation methodology.

## Repository layout

```text
app.py                 Flask application and API endpoint
recommender.py         Retrieval, ranking, filtering, and explanations
build_index.py         Offline FAISS index builder
steam_sqlite.py        SQLite catalogue loader
templates/             Web page template
static/                Browser logic and styling
test_*.py              Regression coverage
eval/metrics.py        Reusable offline metric implementations
docs/                  Public architecture and evaluation notes
vector_index/meta.json Index metadata; the binary index is generated locally
```

## Project status

This repository is structured as a complete, reproducible application. The full Steam catalogue and generated model artefacts are intentionally kept outside ordinary Git history because of size and redistribution constraints. They can be supplied through the documented local configuration and build steps.

## License

MIT. See [`LICENSE`](LICENSE).
