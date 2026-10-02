from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark import (
    Query,
    build_bm25,
    chunk_documents,
    evaluate_rankings,
    load_documents,
    reciprocal_rank_fusion,
    search_bm25,
    search_dense,
)

CHUNK_SIZE = 80
CHUNK_OVERLAP = 16
TOP_K = 10
K_VALUES = [1, 3, 5, 10]
DENSE_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
]


def main() -> None:
    documents = load_documents(PROJECT_ROOT / "data" / "sample_corpus.jsonl")
    queries = [
        Query("q-bm25", "How does BM25 rank documents?", ("doc-002",)),
        Query("q-dense", "What is semantic vector search?", ("doc-003",)),
        Query("q-hybrid", "How can I combine different retrievers?", ("doc-004",)),
        Query("q-metrics", "Which metric credits a relevant result near the top?", ("doc-006",)),
        Query("q-chunks", "Why use overlap between chunks?", ("doc-005",)),
    ]
    chunks = chunk_documents(documents, CHUNK_SIZE, CHUNK_OVERLAP)
    bm25 = build_bm25(chunks)
    lexical_rankings, lexical_latency_ms = search_bm25(bm25, chunks, queries, TOP_K)
    records = [
        {
            "method": "lexical",
            "model": "BM25",
            "query_latency_ms": lexical_latency_ms,
            "index_size_bytes": sum(len(chunk.text.encode("utf-8")) for chunk in chunks),
            **evaluate_rankings(queries, lexical_rankings, K_VALUES),
        }
    ]
    dense_rankings_by_model: dict[str, dict[str, list[str]]] = {}
    for model_name in DENSE_MODELS:
        rankings, latency_ms, index_size_bytes = search_dense(model_name, chunks, queries, TOP_K)
        dense_rankings_by_model[model_name] = rankings
        records.append(
            {
                "method": "dense",
                "model": model_name,
                "query_latency_ms": latency_ms,
                "index_size_bytes": index_size_bytes,
                **evaluate_rankings(queries, rankings, K_VALUES),
            }
        )
    results = pd.DataFrame(records).sort_values(["ndcg@10", "mrr"], ascending=False)
    best_dense_model = results.loc[results["method"].eq("dense"), "model"].iloc[0]
    hybrid_rankings = {
        query.id: reciprocal_rank_fusion(
            [lexical_rankings[query.id], dense_rankings_by_model[best_dense_model][query.id]], TOP_K
        )
        for query in queries
    }
    dense_row = results.loc[results["model"].eq(best_dense_model)].iloc[0]
    results = pd.concat(
        [
            results,
            pd.DataFrame(
                [
                    {
                        "method": "hybrid",
                        "model": f"BM25 + {best_dense_model} (RRF)",
                        "query_latency_ms": lexical_latency_ms + dense_row["query_latency_ms"],
                        "index_size_bytes": int(dense_row["index_size_bytes"]),
                        **evaluate_rankings(queries, hybrid_rankings, K_VALUES),
                    }
                ]
            ),
        ],
        ignore_index=True,
    ).sort_values(["ndcg@10", "mrr"], ascending=False)
    results_path = PROJECT_ROOT / "results" / "benchmark-results.json"
    report_path = PROJECT_ROOT / "results" / "tradeoff-report.md"
    results_path.write_text(results.to_json(orient="records", indent=2))
    best = results.iloc[0]
    report_path.write_text(
        "# Retrieval Benchmark Trade-off Report\n\n"
        f"Best configuration on this dataset: **{best['method']} / {best['model']}**.\n\n"
        "## Results\n\n"
        f"{results.to_markdown(index=False)}\n\n"
        "## Interpretation\n\n"
        "- BM25 is a low-cost lexical baseline and should remain in every comparison.\n"
        "- Dense retrieval captures semantic matches but adds model download, embedding build, and vector storage costs.\n"
        "- Hybrid reciprocal-rank fusion can improve recall when lexical and dense rankings make different errors.\n"
        "- The fixture corpus is a smoke test only; use versioned domain queries and graded qrels before making product decisions.\n"
    )
    print(results.to_string(index=False))
    print(f"Wrote {results_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
