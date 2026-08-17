"""Metadata filter model for parameterized retrieval queries."""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class MetadataFilter(BaseModel):
    """Metadata constraints to filter retrieval candidates."""
    tickers: Optional[List[str]] = Field(default=None, description="Filter by company tickers (e.g. ['NVDA', 'AAPL'])")
    document_types: Optional[List[str]] = Field(default=None, description="Filter by filing types (e.g. ['10-K'])")
    sections: Optional[List[str]] = Field(default=None, description="Filter by section names or codes")
    sectors: Optional[List[str]] = Field(default=None, description="Filter by sectors (e.g. ['Semiconductors'])")
    geographies: Optional[List[str]] = Field(default=None, description="Filter by geographies (e.g. ['US'])")
    fiscal_periods: Optional[List[str]] = Field(default=None, description="Filter by fiscal period strings")
    filing_date_start: Optional[str] = Field(default=None, description="Start date (inclusive, YYYY-MM-DD)")
    filing_date_end: Optional[str] = Field(default=None, description="End date (inclusive, YYYY-MM-DD)")

    def is_empty(self) -> bool:
        """Check whether any filter condition is specified."""
        return not any([
            self.tickers,
            self.document_types,
            self.sections,
            self.sectors,
            self.geographies,
            self.fiscal_periods,
            self.filing_date_start,
            self.filing_date_end,
        ])
