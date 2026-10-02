# Embedding and Retrieval Benchmark

Compare dense, lexical, and hybrid retrieval on a fixed corpus, measuring retrieval quality, latency, and index size.

## What it measures

- Recall@k, MRR, and nDCG for `k = 1, 3, 5, 10`.
- Dense retrieval with multiple Sentence Transformers models.
- Lexical BM25 retrieval.
- Hybrid retrieval using reciprocal-rank fusion.
- Query latency, build latency, and on-disk index size.
- Chunk size, chunk overlap, and top-k as controlled parameters.

## Quick start

```bash
uv sync
uv run python scripts/run_benchmark.py
uv run jupyter lab
```

The first run downloads the configured embedding models. The terminal runner and the notebook both write machine-readable output to `results/benchmark-results.json` and a Markdown summary to `results/tradeoff-report.md`.

## Baseline corpus

`data/sample_corpus.jsonl` is a compact checked-in fixture designed for repeatable smoke tests. Replace it with a versioned domain corpus plus qrels when moving to a meaningful evaluation. The expected JSONL schema is:

```json
{"id": "doc-001", "text": "..."}
```

The notebook includes a small qrels mapping. For a larger benchmark, add `data/queries.jsonl` and `data/qrels.jsonl` and extend `load_evaluation_set` in `src/benchmark.py`.

## Initial model matrix

| Method | Configuration | Why it is here |
| --- | --- | --- |
| Lexical | BM25 | Strong sparse baseline; no model download. |
| Dense | `sentence-transformers/all-MiniLM-L6-v2` | Fast, small general-purpose baseline. |
| Dense | `BAAI/bge-small-en-v1.5` | Popular compact retrieval model. |
| Hybrid | BM25 + best dense result | Measures quality/latency trade-off of reciprocal-rank fusion. |

The matrix is deliberately compact. Add multilingual, domain-specific, or larger GPU models only after the fixed corpus and relevance judgments are reliable.
