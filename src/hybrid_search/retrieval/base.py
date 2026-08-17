"""Base class and common interfaces for all 5 retrieval configurations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence
from qdrant_client import QdrantClient, models
from hybrid_search.models.filter import MetadataFilter
from hybrid_search.models.search import SearchResult


from hybrid_search.retrieval.extractor import AnswerExtractor


class BaseRetriever(ABC):
    """Abstract base class for all Qdrant-backed retrievers."""

    def __init__(self, client: QdrantClient, collection_name: str = "financial_docs"):
        self.client = client
        self.collection_name = collection_name

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for retrieval method."""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[MetadataFilter] = None,
        prefetch_k: int = 50,
    ) -> List[SearchResult]:
        """Executes retrieval against Qdrant collection and returns standardized results."""
        pass

    def _points_to_search_results(
        self,
        points: Sequence[models.ScoredPoint],
        method_name: str,
        query: str = "",
    ) -> List[SearchResult]:
        """Converts raw Qdrant ScoredPoint objects into canonical SearchResult models with extractive answers."""
        results: List[SearchResult] = []
        for rank_idx, pt in enumerate(points, start=1):
            payload = pt.payload or {}
            chunk_text = str(payload.get("text", ""))

            # Extract exact answer & visual highlight if query provided
            exact_answer = ""
            highlighted_text = chunk_text
            confidence = 0.0
            if query:
                extracted = AnswerExtractor.extract(query, chunk_text)
                exact_answer = extracted.exact_answer
                highlighted_text = extracted.highlighted_text
                confidence = extracted.confidence

            results.append(
                SearchResult(
                    chunk_id=str(payload.get("chunk_id", str(pt.id))),
                    document_id=str(payload.get("document_id", "")),
                    score=float(pt.score),
                    rank=rank_idx,
                    text=chunk_text,
                    company=str(payload.get("company", "Unknown")),
                    ticker=str(payload.get("ticker", "")),
                    document_type=str(payload.get("document_type", "")),
                    filing_date=str(payload.get("filing_date", "")),
                    fiscal_period=str(payload.get("fiscal_period", "")),
                    section=str(payload.get("section", "")),
                    sector=str(payload.get("sector", "Unknown")),
                    geography=str(payload.get("geography", "US")),
                    source_url=str(payload.get("source_url", "")),
                    retrieval_method=method_name,
                    exact_answer=exact_answer,
                    highlighted_text=highlighted_text,
                    answer_confidence=confidence,
                )
            )
        return results

