# Evaluation Overview

The project uses separate checks for retrieval, ranking, generation, and reliability.

## Automated checks

- Preference tests cover free/paid, platform, recency, and play-mode interpretation.
- Ranking tests cover reciprocal-rank fusion, review-aware quality, score blending, and diversity constraints.
- Generation tests cover schema validation, duplicate and invalid IDs, retries, partial output, and fallback behavior.
- Reliability tests cover hard-filter allow-lists, reranker degradation, keyword fallback, and safe API errors.
- Offline metric tests cover Recall@k, Precision@k, MRR, nDCG, diversity, and summary statistics.

Run all focused checks with:

```bash
uv run python test_preferences.py
uv run python test_ranking.py
uv run python test_generation.py
uv run python test_reliability.py
uv run python eval/test_metrics.py
```

The detailed experimental judge packets, scoring keys, and intermediate logs are maintained outside the public application surface.
