"""Unit tests for IR evaluation metrics."""

import pytest
from hybrid_search.evaluation.metrics import IRMetrics


def test_recall_at_k():
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    relevant = ["doc2", "doc4", "doc5"]

    # In top 2: doc2 is retrieved -> 1/3
    assert IRMetrics.recall_at_k(retrieved, relevant, k=2) == pytest.approx(1 / 3)
    # In top 4: doc2 and doc4 are retrieved -> 2/3
    assert IRMetrics.recall_at_k(retrieved, relevant, k=4) == pytest.approx(2 / 3)


def test_precision_at_k():
    retrieved = ["doc1", "doc2", "doc3", "doc4"]
    relevant = ["doc2", "doc4"]

    assert IRMetrics.precision_at_k(retrieved, relevant, k=2) == pytest.approx(1 / 2)
    assert IRMetrics.precision_at_k(retrieved, relevant, k=4) == pytest.approx(2 / 4)


def test_reciprocal_rank():
    # First relevant at rank 1
    assert IRMetrics.reciprocal_rank(["doc1", "doc2"], ["doc1"]) == 1.0
    # First relevant at rank 2
    assert IRMetrics.reciprocal_rank(["doc2", "doc1"], ["doc1"]) == 0.5
    # No relevant doc
    assert IRMetrics.reciprocal_rank(["doc2", "doc3"], ["doc1"]) == 0.0


def test_hit_rate_at_k():
    retrieved = ["doc1", "doc2", "doc3"]
    assert IRMetrics.hit_rate_at_k(retrieved, ["doc3"], k=2) == 0.0
    assert IRMetrics.hit_rate_at_k(retrieved, ["doc3"], k=3) == 1.0


def test_ndcg_at_k():
    # Perfect ranking: relevant doc at rank 1
    retrieved_perfect = ["doc1", "doc2"]
    rel_map = {"doc1": 1.0, "doc2": 0.0}
    assert IRMetrics.ndcg_at_k(retrieved_perfect, rel_map, k=2) == pytest.approx(1.0)

    # Suboptimal ranking: relevant doc at rank 2
    retrieved_suboptimal = ["doc2", "doc1"]
    ndcg = IRMetrics.ndcg_at_k(retrieved_suboptimal, rel_map, k=2)
    assert ndcg < 1.0
    assert ndcg > 0.0
