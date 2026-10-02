# Retrieval Benchmark Trade-off Report

Best configuration on this dataset: **dense / sentence-transformers/all-MiniLM-L6-v2**.

## Results

| method   | model                                               |   query_latency_ms |   index_size_bytes |   recall@1 |   ndcg@1 |   recall@3 |   ndcg@3 |   recall@5 |   ndcg@5 |   recall@10 |   ndcg@10 |      mrr |
|:---------|:----------------------------------------------------|-------------------:|-------------------:|-----------:|---------:|-----------:|---------:|-----------:|---------:|------------:|----------:|---------:|
| dense    | sentence-transformers/all-MiniLM-L6-v2              |           2.91626  |              15360 |        1   |      1   |        1   | 1        |        1   | 1        |         1   |  1        | 1        |
| dense    | BAAI/bge-small-en-v1.5                              |           2.23279  |              15360 |        1   |      1   |        1   | 1        |        1   | 1        |         1   |  1        | 1        |
| hybrid   | BM25 + sentence-transformers/all-MiniLM-L6-v2 (RRF) |           3.18873  |              15360 |        0   |      0   |        1   | 0.578558 |        1   | 0.578558 |         1   |  0.578558 | 0.433333 |
| lexical  | BM25                                                |           0.272467 |               1097 |        0.2 |      0.2 |        0.2 | 0.2      |        0.2 | 0.2      |         0.2 |  0.2      | 0.2      |

## Interpretation

- BM25 is a low-cost lexical baseline and should remain in every comparison.
- Dense retrieval captures semantic matches but adds model download, embedding build, and vector storage costs.
- Hybrid reciprocal-rank fusion can improve recall when lexical and dense rankings make different errors.
- The fixture corpus is a smoke test only; use versioned domain queries and graded qrels before making product decisions.
