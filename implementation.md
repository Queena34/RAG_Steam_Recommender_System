# Implementation Summary

## 1. System Architecture Overview

The project is a RAG-based (Retrieval-Augmented Generation) Steam game recommendation system named **raglooker**. It is served as a Flask web application and consists of four main components:

- **`app.py`** — Flask web server. Exposes two routes: `GET /` (renders the frontend) and `POST /api/search` (accepts a natural-language query and returns JSON recommendations). It initialises a single `GameSearchEngine` instance at startup and delegates all search logic to it.

- **`recommender.py`** — Core recommendation engine (`GameSearchEngine`). Implements the full RAG pipeline: query expansion, candidate retrieval, ranking, and answer generation. Also manages initialisation of all sub-components (FAISS index, embedding function, LLM, BM25 index).

- **`build_index.py`** — Offline index builder. Reads game records from SQLite, constructs a rich text document for each game, embeds them via Ollama, and stores the resulting vectors in a FAISS index on disk (`vector_index/`). Must be run once before starting the Flask app.

- **`steam_sqlite.py`** — Utility module for loading raw game records from the SQLite database into Python dictionaries.

---

## 2. Data Flow

```
User query (natural language)
        │
        ▼
[Query Expansion]  ── LLM (Ollama) generates 8–10 related keywords/tags
        │
        ▼
[Candidate Retrieval]  ── selects top-50 candidate games:
        │   1. FAISS vector search and BM25 lexical search each return top-100
        │   2. Reciprocal Rank Fusion (RRF) merges both ranked lists
        │   3. SQLite LIKE search is used only if neither index is available
        │
        ▼
[Re-scoring]  ── blends normalized RRF relevance with Wilson review quality
        │
        ▼
[Ranking]  ── applies explicit free/platform filters, preference bonuses,
        │      optional ONNX Cross-encoder, quality ranking and diversity; keeps top-10
        │
        ▼
[Answer Generation]  ── schema-constrained LLM selects up to five app_ids and
        │              writes reasons/evidence; invalid output is retried or falls back
        │
        ▼
JSON response  →  { matches: [...], answer: "...", meta: {...} }
```

---

## 3. Core Functions

### `retrieve_candidates(query)`

Expands the user query with LLM-generated keywords, runs both available retrievers, fuses their ranked IDs with RRF, and hydrates the best 50 candidates:

1. **FAISS vector search** (`_retrieve_vector_ids`): embeds the expanded query with the query prefix stored in `vector_index/meta.json`, normalizes it, and searches the 768-dimensional inner-product index.
2. **BM25 lexical search** (`_retrieve_bm25_ids`): tokenizes with a Porter stemmer and searches a cached corpus containing names, descriptions, tags, genres, and the same sampled review text used by the vector corpus.
3. **RRF fusion** (`_rrf_fuse`): merges the two top-100 ID lists using `1 / (k + rank)`, with `k=60`, avoiding incompatible raw-score scales.
4. **SQLite LIKE fallback** (`_retrieve_keyword`): performs AND-then-OR keyword matching across game name, description, tags, and genres only when both indexed retrievers fail.

### `rank_candidates(query, candidates)`

Sorts candidates by the pre-computed score plus small explicit-preference bonuses, applies conditional free/platform filters, then applies developer/tag diversity constraints and returns up to 10 candidates. The generative `_rank_with_llm()` implementation remains in the file but is not called at runtime because it takes roughly 550 seconds per query on the target CPU. The final LLM generation step selects the displayed five when it succeeds.

### `generate_answer(query, matches)`

Calls structured generation inside a 360-second timeout thread. The LLM receives up to 10 ranked candidates with app IDs, names, tags, ratings, and short descriptions. Ollama enforces a schema containing five recommendation slots; the code validates IDs against the candidate set, removes duplicates, measures evidence overlap, retries once at lower temperature when necessary, and tops up from the ranked list when the output is partial. If generation fails or times out, `_simple_answer` returns a plain-text fallback.

---

## 4. Tech Stack

| Component | Library / Tool | Role |
|---|---|---|
| Web framework | Flask 3.x | HTTP server and routing |
| Vector index | FAISS (`faiss-cpu`) | Exact inner-product search over normalized game embeddings |
| Embedding model | `nomic-embed-text` via Ollama | Converts text to 768-d vectors for semantic search |
| LLM | `phi4-mini` (or any available Ollama model) | Query intent parsing, query expansion and answer generation |
| Cross-encoder | Optional ONNX Runtime + `tokenizers` | Reranks up to 50 fused candidates; falls back when unavailable |
| Lexical search | `rank-bm25` + NLTK | Cached BM25Okapi corpus, run in parallel with FAISS |
| Database | SQLite (`steam_games_reviews_25.sqlite`) | Stores 39,176 game records and 7,679,845 reviews |
| Numerical compute | NumPy | Vector normalisation and score blending |
| Progress display | tqdm | Index build progress bar |

---

## 5. Index Building Process (`build_index.py`)

The index is built offline and only needs to be run once (or resumed if interrupted):

1. **Connect to Ollama** — verifies that `nomic-embed-text` is reachable and records the embedding dimension.
2. **Resume support** — if `vector_index/index.faiss` and `vector_index/meta.json` already exist, the builder loads them and skips already-indexed games.
3. **Fetch games in batches** — queries games with `positive > 10` from SQLite in batches of 100, filtering out already-indexed IDs.
4. **Build document text** — for each game, constructs a rich document containing: game name, short description, genres, top-15 tags, whitelisted gameplay modes, release date, rating percentage, six helpful positive-review snippets and three negative-review snippets.
5. **Embed in sub-batches** — sends documents to Ollama in sub-batches of 50 for embedding.
6. **L2-normalise and add to FAISS** — vectors are L2-normalised (enabling inner-product search to behave as cosine similarity) before being added to a `faiss.IndexFlatIP` index.
7. **Checkpoint after every batch** — atomically saves the FAISS index and a `meta.json` file storing `app_ids`, model, dimension, document/query prefixes, and the coverage threshold so the build can be safely interrupted and resumed.

The resulting `vector_index/` directory contains:
- `index.faiss` — the serialised FAISS index
- `meta.json` — ordered list of `app_ids` and index metadata

To enable Cross-encoder reranking, provide a pre-exported ONNX model directory containing `model.onnx` and `tokenizer.json`, then set `RAGLOOKER_RERANKER_MODEL_DIR`. The API metadata exposes `reranker_status` and `reranker_applied` so a missing model cannot fail silently.

---

## 6. 执行步骤
1. 安装依赖
uv sync

2. 确保Ollama已启动并拉取模型
ollama pull nomic-embed-text
ollama pull phi4-mini

3. 把 sqlite 数据库文件放入项目根目录，或设置 `RAGLOOKER_DB_PATH`

4. 建立FAISS索引（需要几个小时）
uv run python build_index.py

5. 启动Flask
uv run flask --app app run --debug

---
## 7. 测试结果
查询：relaxing farming or life simulation game
返回：Stardew Valley，Farm Together，Farming Simulator 17，Rusty's Retirement，Coral Island

chtgpt查询结果：
Best overall: Stardew Valley
Classic farm-life loop: crops, animals, fishing, mining, relationships, and town life. It is still one of the strongest “cozy but deep” options on Steam, and it continues to get quality-of-life updates.

Most chill / low-stress: Farm Together 2
Great if you want something more soothing and less demanding. Its Steam page leans hard into the “cozy, relaxing farming experience,” and it supports solo or co-op play.

Best if you want community vibes: Roots of Pacha
A relaxed farming sim with a prehistoric setting, village-building, animal friendship, and strong community focus. Good choice if you like Stardew-style structure but want something fresher and gentler.

Best if you want modern cozy + relationships: Coral Island
Laid-back island farming with relationships, animals, and a broader town/social feel. It is especially good if you want a polished, colorful life-sim with lots of character interaction.

Best if you like cozy with a little consequence: Echoes of the Plum Grove
Still cozy, but with more survival-ish decisions and seasonal pressure. Recommended if you want a farm sim that feels warm and charming without being totally frictionless.

A few more worth a look:
Ooblets for cute and goofy collecting, Spiritfarer if you want a more emotional life-management game than a pure farm sim, and Yonder: The Cloud Catcher Chronicles if you want exploration with very low stress. Steam’s farming and life-sim categories also surface a lot of adjacent cozy options.



---

查询：scary horror survival game
返回：The Forest，Project Zomboid，Sons Of The Forest，Alien: Isolation，The Evil Within

chatgpt推荐：
Best overall: Resident Evil 4
A great pick if you want survival horror with strong combat, pressure, resource management, and constant tension rather than pure helplessness. The Steam page explicitly frames it as a modernized survival-horror classic.

Scariest “I can barely fight back” pick: Alien: Isolation
This is the one I’d recommend if you want dread, stealth, and being hunted. Steam describes it as “constant dread and mortal danger,” with survival built around scavenging and using your wits.

Best pure survival-horror design: Amnesia: The Bunker
Excellent if you want fear driven by systems: limited tools, darkness, improvisation, and a relentless AI monster. Its Steam page emphasizes crafting, resource use, and keeping the lights on to stay alive.

Best co-op horror survival: The Outlast Trials
Go here if you want to panic with friends. It is built around surviving terrifying “trials” solo or in teams, and the Steam page highlights that co-op angle directly.

Best action-heavy sci-fi survival horror: Dead Space
If you want more fighting but still a strong horror atmosphere, this is a top choice. The remake is positioned on Steam as a rebuilt survival-horror classic with major visual, audio, and gameplay improvements.

Best open-world survival with horror flavor: Sons Of The Forest
This is less “tight corridor terror” and more “craft, build, and survive in a horrifying wilderness.” Best if you want survival systems first and horror second.




