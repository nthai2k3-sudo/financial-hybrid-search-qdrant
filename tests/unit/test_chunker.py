"""Unit tests for DocumentChunker."""

from hybrid_search.chunking.chunker import DocumentChunker
from hybrid_search.models.document import Document, DocumentSection


def test_chunk_document_basic(sample_document: Document):
    chunker = DocumentChunker(chunk_size_tokens=100, chunk_overlap_tokens=20)
    chunks = chunker.chunk_document(sample_document)
    assert len(chunks) >= 1
    c = chunks[0]
    assert c.document_id == sample_document.document_id
    assert c.ticker == "NVDA"
    assert c.chunk_id == f"{sample_document.document_id}_chunk_0000"
    assert c.section == "Item 1A: Risk Factors"
    assert len(c.text_hash) == 64


def test_chunking_determinism(sample_document: Document):
    chunker = DocumentChunker(chunk_size_tokens=200, chunk_overlap_tokens=50)
    chunks1 = chunker.chunk_document(sample_document)
    chunks2 = chunker.chunk_document(sample_document)

    assert len(chunks1) == len(chunks2)
    for c1, c2 in zip(chunks1, chunks2):
        assert c1.chunk_id == c2.chunk_id
        assert c1.text == c2.text
        assert c1.text_hash == c2.text_hash


def test_chunking_large_text():
    long_text = " ".join([f"Sentence number {i} provides financial disclosure on operations." for i in range(100)])
    doc = Document(
        document_id="doc_large",
        company="Large Corp",
        ticker="LRG",
        cik="0001999999",
        accession_number="0001999999-24-000001",
        document_type="10-K",
        filing_date="2024-03-01",
        report_date="2023-12-31",
        fiscal_period="FY2023",
        source_url="https://example.com",
        sections=[
            DocumentSection(section_name="Item 7: MD&A", section_code="ITEM_7", text=long_text)
        ],
    )
    chunker = DocumentChunker(chunk_size_tokens=50, chunk_overlap_tokens=10)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) > 1
    assert chunks[0].chunk_id == "doc_large_chunk_0000"
    assert chunks[1].chunk_id == "doc_large_chunk_0001"
