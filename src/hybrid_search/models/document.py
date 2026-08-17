"""Document and Chunk data models for the financial search pipeline."""

from __future__ import annotations

import hashlib
from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentSection(BaseModel):
    """Represents a discrete narrative section extracted from an SEC filing."""
    section_name: str = Field(..., description="Readable section title, e.g. 'Item 1A: Risk Factors'")
    section_code: str = Field(..., description="Canonical code, e.g. 'ITEM_1', 'ITEM_1A', 'ITEM_7', 'ITEM_2'")
    text: str = Field(..., description="Extracted raw narrative text")
    word_count: int = Field(default=0, description="Word count of section text")

    def model_post_init(self, __context: object) -> None:
        if not self.word_count and self.text:
            self.word_count = len(self.text.split())


class Document(BaseModel):
    """Represents an ingested SEC filing document."""
    document_id: str = Field(..., description="Deterministic ID: f'{cik}_{accession_number}'")
    company: str = Field(..., description="Company name, e.g. 'NVIDIA Corporation'")
    ticker: str = Field(..., description="Ticker symbol, e.g. 'NVDA'")
    cik: str = Field(..., description="SEC CIK number, e.g. '0001045810'")
    accession_number: str = Field(..., description="SEC filing accession number")
    document_type: str = Field(..., description="Filing type: '10-K', '10-Q', '20-F', '6-K'")
    filing_date: str = Field(..., description="Filing date in YYYY-MM-DD format")
    report_date: str = Field(..., description="Report period end date in YYYY-MM-DD format")
    fiscal_period: str = Field(..., description="Fiscal quarter/year, e.g. 'FY2024' or 'Q3 FY2024'")
    source_url: str = Field(..., description="Direct SEC EDGAR archive URL")
    sections: List[DocumentSection] = Field(default_factory=list, description="Extracted filing sections")
    sector: Optional[str] = Field(default=None, description="Industry sector")
    geography: Optional[str] = Field(default=None, description="Primary geographic market / headquarters")


class Chunk(BaseModel):
    """Represents a text chunk with complete metadata and vector payload slots."""
    chunk_id: str = Field(..., description="Deterministic ID: f'{document_id}_chunk_{chunk_index:04d}'")
    document_id: str = Field(..., description="Parent filing document ID")
    company: str = Field(..., description="Company name")
    ticker: str = Field(..., description="Ticker symbol")
    cik: str = Field(..., description="SEC CIK number")
    document_type: str = Field(..., description="Filing type ('10-K', '10-Q', '20-F')")
    filing_date: str = Field(..., description="Filing date YYYY-MM-DD")
    fiscal_period: str = Field(..., description="Fiscal period")
    section: str = Field(..., description="Section title or code")
    sector: str = Field(default="Unknown", description="Industry sector")
    geography: str = Field(default="US", description="Geographic jurisdiction")
    text: str = Field(..., description="Chunk text content")
    text_hash: str = Field(default="", description="SHA-256 hash of text content")
    chunk_index: int = Field(default=0, description="0-indexed position within parent document")
    source_url: str = Field(default="", description="EDGAR source URL")
    accession_number: str = Field(default="", description="Parent accession number")

    # Vector representations populated during embedding / indexing
    dense_vector: Optional[List[float]] = Field(default=None, description="Dense embedding (e.g. 384 dims)")
    bm25_indices: Optional[List[int]] = Field(default=None, description="Qdrant/bm25 sparse vector indices")
    bm25_values: Optional[List[float]] = Field(default=None, description="Qdrant/bm25 sparse vector values")
    sparse_indices: Optional[List[int]] = Field(default=None, description="SPLADE sparse vector indices")
    sparse_values: Optional[List[float]] = Field(default=None, description="SPLADE sparse vector values")

    def model_post_init(self, __context: object) -> None:
        if not self.text_hash and self.text:
            self.text_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
