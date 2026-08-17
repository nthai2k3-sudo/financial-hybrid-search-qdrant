"""SEC EDGAR API client for discovering company filing metadata."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class FilingMetadata(BaseModel):
    """Metadata representing a single SEC filing available for download."""
    ticker: str
    company_name: str
    cik: str
    form: str
    filing_date: str
    report_date: str
    accession_number: str
    primary_document: str
    document_url: str
    fiscal_period: str = "Unknown"
    sector: str = "Unknown"
    geography: str = "US"


class EdgarClient:
    """Discovers SEC filings via the SEC Submissions API."""

    SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik10}.json"
    ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_no_dashes}/{primary_doc}"

    @classmethod
    def format_cik(cls, cik: str | int) -> str:
        """Formats CIK as 10-digit zero-padded string."""
        return str(cik).strip().zfill(10)

    @classmethod
    def build_document_url(cls, cik: str, accession_number: str, primary_document: str) -> str:
        """Constructs direct archive URL for a primary filing document."""
        cik_int = int(str(cik).strip())
        acc_no_dashes = accession_number.replace("-", "")
        return cls.ARCHIVES_BASE_URL.format(
            cik_int=cik_int,
            acc_no_dashes=acc_no_dashes,
            primary_doc=primary_document,
        )

    @classmethod
    def parse_submissions_json(
        cls,
        submissions: Dict[str, Any],
        ticker: str,
        company_name: str,
        target_forms: Optional[List[str]] = None,
        start_year: int = 2023,
        end_year: int = 2025,
        max_10q_per_company: int = 2,
        sector: str = "Unknown",
        geography: str = "US",
    ) -> List[FilingMetadata]:
        """Parses EDGAR submissions payload and extracts target filings."""
        cik_raw = submissions.get("cik", "")
        cik = cls.format_cik(cik_raw) if cik_raw else ""
        recent = submissions.get("filings", {}).get("recent", {})
        if not recent:
            return []

        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        report_dates = recent.get("reportDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])

        num_records = len(forms)
        filings: List[FilingMetadata] = []
        target_forms_set = set(target_forms or ["10-K", "10-Q", "20-F"])

        count_10q = 0

        for i in range(num_records):
            form = forms[i]
            if form not in target_forms_set:
                continue

            filing_date = filing_dates[i] if i < len(filing_dates) else ""
            report_date = report_dates[i] if i < len(report_dates) else filing_date
            try:
                year = int(filing_date.split("-")[0]) if filing_date else 0
            except ValueError:
                year = 0

            if year < start_year or year > end_year:
                continue

            # Scope limit: max 2 recent 10-Q / 6-K filings per company to keep corpus compact
            if form in ("10-Q", "6-K"):
                if count_10q >= max_10q_per_company:
                    continue
                count_10q += 1

            acc_num = accessions[i] if i < len(accessions) else ""
            primary_doc = primary_docs[i] if i < len(primary_docs) else ""
            doc_url = cls.build_document_url(cik, acc_num, primary_doc)

            # Determine fiscal period indicator
            fiscal_period = f"FY{year}" if form in ("10-K", "20-F") else f"Q-{filing_date[:7]}"

            filings.append(
                FilingMetadata(
                    ticker=ticker,
                    company_name=company_name,
                    cik=cik,
                    form=form,
                    filing_date=filing_date,
                    report_date=report_date,
                    accession_number=acc_num,
                    primary_document=primary_doc,
                    document_url=doc_url,
                    fiscal_period=fiscal_period,
                    sector=sector,
                    geography=geography,
                )
            )

        return filings
