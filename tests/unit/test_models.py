"""Unit tests for Pydantic data models."""

from hybrid_search.models.document import Chunk, Document, DocumentSection
from hybrid_search.models.filter import MetadataFilter
from hybrid_search.models.search import SearchQuery, SearchResult


def test_document_section_initialization():
    sec = DocumentSection(
        section_name="Item 1: Business",
        section_code="ITEM_1",
        text="NVIDIA is the pioneer of GPU-accelerated computing.",
    )
    assert sec.word_count == 7
    assert sec.section_code == "ITEM_1"


def test_document_serialization(sample_document: Document):
    data = sample_document.model_dump()
    assert data["ticker"] == "NVDA"
    assert data["document_id"] == "0001045810_0001045810-24-000029"
    assert len(data["sections"]) == 1

    restored = Document.model_validate(data)
    assert restored.company == sample_document.company


def test_chunk_hash_generation():
    chunk = Chunk(
        chunk_id="test_01",
        document_id="doc_01",
        company="Test Co",
        ticker="TST",
        cik="0001234567",
        document_type="10-K",
        filing_date="2024-01-01",
        fiscal_period="FY2023",
        section="Item 1",
        text="Unique test chunk text.",
    )
    assert len(chunk.text_hash) == 64  # SHA-256 hex string


def test_metadata_filter_is_empty():
    f_empty = MetadataFilter()
    assert f_empty.is_empty() is True

    f_active = MetadataFilter(tickers=["NVDA"])
    assert f_active.is_empty() is False


def test_search_result_and_query():
    res = SearchResult(
        chunk_id="chunk_01",
        document_id="doc_01",
        score=0.885,
        rank=1,
        text="Sample retrieval result",
        company="NVIDIA Corporation",
        ticker="NVDA",
        document_type="10-K",
        filing_date="2024-02-21",
        fiscal_period="FY2024",
        section="Item 1A",
        retrieval_method="sparse_dense",
    )
    assert res.score == 0.885
    assert res.rank == 1

    query = SearchQuery(query_text="NVIDIA H200 supply", k=5, method="sparse_dense")
    assert query.k == 5
    assert query.prefetch_k == 50
