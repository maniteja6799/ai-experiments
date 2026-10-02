from src.benchmark import Query, evaluate_rankings, ndcg_at_k, reciprocal_rank


def test_ranking_metrics_reward_relevant_first_result() -> None:
    relevant = {"doc-1"}
    assert reciprocal_rank(["doc-1", "doc-2"], relevant) == 1.0
    assert ndcg_at_k(["doc-1", "doc-2"], relevant, 2) == 1.0


def test_evaluate_rankings_averages_queries() -> None:
    queries = [
        Query(id="q1", text="one", relevant_ids=("doc-1",)),
        Query(id="q2", text="two", relevant_ids=("doc-2",)),
    ]
    metrics = evaluate_rankings(queries, {"q1": ["doc-1"], "q2": ["doc-3"]}, [1])
    assert metrics == {"recall@1": 0.5, "ndcg@1": 0.5, "mrr": 0.5}
