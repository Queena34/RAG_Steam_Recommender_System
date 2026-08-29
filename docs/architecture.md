# Architecture Overview

The application has an offline preparation stage and an online recommendation stage.

## Offline stage

`build_index.py` reads the Steam catalogue from SQLite, constructs one searchable document per eligible game, embeds the documents with Ollama, and writes a FAISS index plus metadata. The generated binary index is intentionally not tracked in the public repository; it can be rebuilt from the configured catalogue.

## Online stage

1. The Flask API validates the natural-language query.
2. Optional structured intent parsing extracts hard constraints such as platform, price, and play mode.
3. FAISS and BM25 retrieve complementary candidate sets when available.
4. Reciprocal-rank fusion combines the retrieval lists.
5. Hard filters, preference bonuses, review-aware quality, and diversity constraints produce a ranked shortlist.
6. An optional ONNX Cross-encoder reranks the shortlist.
7. Structured generation selects valid catalogue IDs and writes grounded reasons. If generation is unavailable, database-backed fallback explanations are used.

The API exposes operational metadata so a degraded retrieval or generation path is visible to callers rather than silently changing behavior.
