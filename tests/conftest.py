"""Pytest fixtures shared across test suites."""

from __future__ import annotations

from typing import List
import pytest
from hybrid_search.models.document import Chunk, Document, DocumentSection
from hybrid_search.models.filter import MetadataFilter
from hybrid_search.models.search import SearchResult


@pytest.fixture
def sample_section() -> DocumentSection:
    return DocumentSection(
        section_name="Item 1A: Risk Factors",
        section_code="ITEM_1A",
        text="Our future operating results may fluctuate due to supply chain constraints on advanced AI GPUs.",
    )


@pytest.fixture
def sample_document(sample_section: DocumentSection) -> Document:
    return Document(
        document_id="0001045810_0001045810-24-000029",
        company="NVIDIA Corporation",
        ticker="NVDA",
        cik="0001045810",
        accession_number="0001045810-24-000029",
        document_type="10-K",
        filing_date="2024-02-21",
        report_date="2024-01-28",
        fiscal_period="FY2024",
        source_url="https://www.sec.gov/Archives/edgar/data/1045810/000104581024000029/nvda-20240128.htm",
        sections=[sample_section],
        sector="Semiconductors",
        geography="US",
    )


@pytest.fixture
def sample_chunks() -> List[Chunk]:
    return [
        Chunk(
            chunk_id="0001045810_0001045810-24-000029_chunk_0000",
            document_id="0001045810_0001045810-24-000029",
            company="NVIDIA Corporation",
            ticker="NVDA",
            cik="0001045810",
            document_type="10-K",
            filing_date="2024-02-21",
            fiscal_period="FY2024",
            section="Item 1A: Risk Factors",
            sector="Semiconductors",
            geography="US",
            text="NVIDIA faces manufacturing capacity constraints at TSMC for H100 and H200 accelerator platforms.",
            chunk_index=0,
            source_url="https://www.sec.gov/Archives/edgar/data/1045810/000104581024000029/nvda-20240128.htm",
            accession_number="0001045810-24-000029",
        ),
        Chunk(
            chunk_id="0001045810_0001045810-24-000029_chunk_0001",
            document_id="0001045810_0001045810-24-000029",
            company="NVIDIA Corporation",
            ticker="NVDA",
            cik="0001045810",
            document_type="10-K",
            filing_date="2024-02-21",
            fiscal_period="FY2024",
            section="Item 7: MD&A",
            sector="Semiconductors",
            geography="US",
            text="Capital expenditure in data center infrastructure expanded significantly across major cloud service providers.",
            chunk_index=1,
            source_url="https://www.sec.gov/Archives/edgar/data/1045810/000104581024000029/nvda-20240128.htm",
            accession_number="0001045810-24-000029",
        ),
        Chunk(
            chunk_id="0000320193_0000320193-24-000106_chunk_0000",
            document_id="0000320193_0000320193-24-000106",
            company="Apple Inc.",
            ticker="AAPL",
            cik="0000320193",
            document_type="10-K",
            filing_date="2024-11-01",
            fiscal_period="FY2024",
            section="Item 7: MD&A",
            sector="Consumer Electronics",
            geography="US",
            text="Share repurchase activity authorization and dividends returned substantial capital to shareholders during fiscal 2024.",
            chunk_index=0,
            source_url="https://www.sec.gov/Archives/edgar/data/320193/000032019324000106/aapl-20240928.htm",
            accession_number="0000320193-24-000106",
        ),
    ]
