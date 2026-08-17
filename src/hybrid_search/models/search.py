"""Search result and query models for standardized retrieval interfaces."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field
from hybrid_search.models.filter import MetadataFilter


class SearchResult(BaseModel):
    """Canonical search result returned uniformly across all 5 retrieval methods."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document identifier")
    score: float = Field(..., description="Relevance score (cosine similarity, BM25 TF-IDF, or RRF score)")
    rank: int = Field(..., description="1-indexed rank in result set")
    text: str = Field(..., description="Text content of the retrieved chunk")
    company: str = Field(..., description="Company name")
    ticker: str = Field(..., description="Ticker symbol")
    document_type: str = Field(..., description="Document type, e.g. 10-K, 10-Q")
    filing_date: str = Field(..., description="Filing date YYYY-MM-DD")
    fiscal_period: str = Field(..., description="Fiscal period")
    section: str = Field(..., description="Section title or code")
    sector: str = Field(default="Unknown", description="Industry sector")
    geography: str = Field(default="US", description="Geographic jurisdiction")
    source_url: str = Field(default="", description="EDGAR source URL")
    retrieval_method: str = Field(
        ...,
        description="Method used: 'bm25', 'dense', 'sparse', 'bm25_dense', 'sparse_dense', 'rerank'"
    )
    exact_answer: str = Field(default="", description="Pinpointed exact answering sentence or table row")
    highlighted_text: str = Field(default="", description="Chunk text with the answering excerpt marked")
    answer_confidence: float = Field(default=0.0, description="Confidence score of extracted answer")



class SearchQuery(BaseModel):
    """Encapsulates a search request with query text, parameters, and filters."""
    query_text: str = Field(..., description="Natural language or keyword query")
    k: int = Field(default=10, ge=1, le=100, description="Top-K results to return")
    method: str = Field(
        default="sparse_dense",
        description="Retrieval method: 'bm25', 'dense', 'bm25_dense', 'sparse', 'sparse_dense'"
    )
    filters: Optional[MetadataFilter] = Field(default=None, description="Optional metadata filter constraints")
    prefetch_k: int = Field(default=50, ge=1, description="Number of candidate points to prefetch for hybrid fusion")
    rrf_constant: int = Field(default=60, ge=1, description="RRF constant (typically 60)")
