from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import bm25s
import numpy as np


@dataclass(frozen=True)
class Document:
    id: str
    text: str


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    relevant_ids: tuple[str, ...]


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    text: str


def load_documents(path: Path) -> list[Document]:
    return [Document(**json.loads(line)) for line in path.read_text().splitlines() if line]


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("chunk size must be positive and overlap must be smaller than size")
    words = text.split()
    return [" ".join(words[start : start + size]) for start in range(0, len(words), size - overlap)]


def chunk_documents(documents: Sequence[Document], size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        for position, text in enumerate(chunk_text(document.text, size, overlap)):
            chunks.append(Chunk(id=f"{document.id}:{position}", document_id=document.id, text=text))
    return chunks


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    return float(bool(set(ranked_ids[:k]) & relevant_ids))


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: set[str]) -> float:
    for position, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    gains = [1.0 / np.log2(position + 2) for position, doc_id in enumerate(ranked_ids[:k]) if doc_id in relevant_ids]
    ideal = sum(1.0 / np.log2(position + 2) for position in range(min(k, len(relevant_ids))))
    return sum(gains) / ideal if ideal else 0.0


def evaluate_rankings(queries: Iterable[Query], rankings: dict[str, Sequence[str]], k_values: Sequence[int]) -> dict[str, float]:
    query_list = list(queries)
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"recall@{k}"] = float(np.mean([recall_at_k(rankings[q.id], set(q.relevant_ids), k) for q in query_list]))
        metrics[f"ndcg@{k}"] = float(np.mean([ndcg_at_k(rankings[q.id], set(q.relevant_ids), k) for q in query_list]))
    metrics["mrr"] = float(np.mean([reciprocal_rank(rankings[q.id], set(q.relevant_ids)) for q in query_list]))
    return metrics


def build_bm25(documents: Sequence[Document | Chunk]):
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize([document.text for document in documents]))
    return retriever


def search_bm25(retriever, documents: Sequence[Document | Chunk], queries: Sequence[Query], top_k: int) -> tuple[dict[str, list[str]], float]:
    started = time.perf_counter()
    _, indices = retriever.retrieve(bm25s.tokenize([query.text for query in queries]), k=top_k)
    elapsed_ms = (time.perf_counter() - started) * 1_000
    rankings = {
        query.id: [getattr(documents[int(index)], "document_id", documents[int(index)].id) for index in indices[row]]
        for row, query in enumerate(queries)
    }
    return rankings, elapsed_ms / len(queries)


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], top_k: int, rrf_k: int = 60) -> list[str]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]]


def collapse_chunk_rankings(chunk_rankings: np.ndarray, chunks: Sequence[Chunk], top_k: int) -> list[str]:
    document_ids: list[str] = []
    for chunk_index in chunk_rankings:
        document_id = chunks[int(chunk_index)].document_id
        if document_id not in document_ids:
            document_ids.append(document_id)
        if len(document_ids) == top_k:
            break
    return document_ids


def search_dense(model_name: str, chunks: Sequence[Chunk], queries: Sequence[Query], top_k: int) -> tuple[dict[str, list[str]], float, int]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    document_embeddings = model.encode([chunk.text for chunk in chunks], normalize_embeddings=True, show_progress_bar=True)
    started = time.perf_counter()
    query_embeddings = model.encode([query.text for query in queries], normalize_embeddings=True, show_progress_bar=False)
    ranked_chunk_indices = np.argsort(-(query_embeddings @ document_embeddings.T), axis=1)
    elapsed_ms = (time.perf_counter() - started) * 1_000
    rankings = {
        query.id: collapse_chunk_rankings(ranked_chunk_indices[row], chunks, top_k)
        for row, query in enumerate(queries)
    }
    return rankings, elapsed_ms / len(queries), int(document_embeddings.nbytes)
