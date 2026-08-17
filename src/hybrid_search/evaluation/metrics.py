"""Information retrieval evaluation metrics."""

from __future__ import annotations

import math
from typing import Dict, List, Set, Union


class IRMetrics:
    """Standardized information retrieval metrics for evaluating search quality."""

    @staticmethod
    def recall_at_k(
        retrieved_ids: List[str],
        relevant_ids: Union[List[str], Set[str]],
        k: int = 10,
    ) -> float:
        """Recall@K = (|Relevant ∩ Retrieved@K|) / |Total Relevant|."""
        rel_set = set(relevant_ids)
        if not rel_set:
            return 0.0
        ret_k = retrieved_ids[:k]
        hits = len(set(ret_k) & rel_set)
        return hits / len(rel_set)

    @staticmethod
    def precision_at_k(
        retrieved_ids: List[str],
        relevant_ids: Union[List[str], Set[str]],
        k: int = 10,
    ) -> float:
        """Precision@K = (|Relevant ∩ Retrieved@K|) / K."""
        if k <= 0:
            return 0.0
        rel_set = set(relevant_ids)
        ret_k = retrieved_ids[:k]
        hits = len(set(ret_k) & rel_set)
        return hits / k

    @staticmethod
    def reciprocal_rank(
        retrieved_ids: List[str],
        relevant_ids: Union[List[str], Set[str]],
    ) -> float:
        """Reciprocal Rank (RR) = 1 / rank of first relevant item (1-indexed)."""
        rel_set = set(relevant_ids)
        if not rel_set:
            return 0.0
        for rank_idx, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in rel_set:
                return 1.0 / rank_idx
        return 0.0

    @staticmethod
    def hit_rate_at_k(
        retrieved_ids: List[str],
        relevant_ids: Union[List[str], Set[str]],
        k: int = 10,
    ) -> float:
        """Hit Rate@K = 1.0 if at least one relevant item in top-K else 0.0."""
        rel_set = set(relevant_ids)
        if not rel_set:
            return 0.0
        ret_k = retrieved_ids[:k]
        return 1.0 if any(doc_id in rel_set for doc_id in ret_k) else 0.0

    @staticmethod
    def ndcg_at_k(
        retrieved_ids: List[str],
        relevance_map: Union[List[str], Set[str], Dict[str, float]],
        k: int = 10,
    ) -> float:
        """Normalized Discounted Cumulative Gain at rank K."""
        if k <= 0 or not relevance_map:
            return 0.0

        # Convert set/list to binary relevance dict if necessary
        if isinstance(relevance_map, (list, set)):
            rel_dict: Dict[str, float] = {doc_id: 1.0 for doc_id in relevance_map}
        else:
            rel_dict = relevance_map

        ret_k = retrieved_ids[:k]
        dcg = 0.0
        for i, doc_id in enumerate(ret_k):
            rel = rel_dict.get(doc_id, 0.0)
            if rel > 0:
                dcg += rel / math.log2(i + 2)  # i+2 for 1-based log2(rank + 1)

        # Ideal DCG
        ideal_rels = sorted(rel_dict.values(), reverse=True)[:k]
        idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels) if rel > 0)

        if idcg == 0.0:
            return 0.0

        return dcg / idcg
