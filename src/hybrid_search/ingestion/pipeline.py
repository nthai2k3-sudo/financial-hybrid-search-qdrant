"""Ingestion pipeline orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from hybrid_search.config import AppConfig, CompanyConfig
from hybrid_search.ingestion.downloader import EdgarDownloader
from hybrid_search.ingestion.edgar_client import EdgarClient, FilingMetadata
from hybrid_search.ingestion.html_parser import EdgarHtmlParser
from hybrid_search.ingestion.normalizer import TextNormalizer
from hybrid_search.ingestion.section_extractor import SectionExtractor
from hybrid_search.models.document import Document


class IngestionPipeline:
    """End-to-end ingestion pipeline from SEC EDGAR to parsed Document models."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.downloader = EdgarDownloader(
            user_agent=config.edgar.user_agent,
            rate_limit_per_second=config.edgar.rate_limit_per_second,
            request_timeout_seconds=config.edgar.request_timeout_seconds,
            max_retries=config.edgar.max_retries,
            cache_dir=config.edgar.cache_dir,
        )

    def discover_company_filings(
        self,
        company: CompanyConfig,
        start_year: int = 2023,
        end_year: int = 2025,
    ) -> List[FilingMetadata]:
        """Fetches submissions metadata for a single company."""
        cik10 = EdgarClient.format_cik(company.cik)
        url = EdgarClient.SUBMISSIONS_URL.format(cik10=cik10)
        submissions = self.downloader.fetch_json(url)
        if not submissions:
            print(f"Warning: Could not fetch submissions for {company.name} ({company.ticker})")
            return []

        return EdgarClient.parse_submissions_json(
            submissions=submissions,
            ticker=company.ticker,
            company_name=company.name,
            target_forms=company.forms,
            start_year=start_year,
            end_year=end_year,
            max_10q_per_company=2,
            sector=company.sector,
            geography=company.geography,
        )

    def process_filing_file(self, file_path: Path, metadata: FilingMetadata) -> Optional[Document]:
        """Parses a local filing HTML file into a structured Document object."""
        try:
            raw_html = file_path.read_text(encoding="utf-8", errors="ignore")
            clean_text = EdgarHtmlParser.parse_html_to_text(raw_html)
            raw_sections = SectionExtractor.extract_sections(clean_text, metadata.form)

            # Normalize text within sections
            normalized_sections = []
            for s in raw_sections:
                s.text = TextNormalizer.normalize(s.text)
                s.word_count = len(s.text.split())
                normalized_sections.append(s)

            doc_id = f"{metadata.cik}_{metadata.accession_number}"
            return Document(
                document_id=doc_id,
                company=metadata.company_name,
                ticker=metadata.ticker,
                cik=metadata.cik,
                accession_number=metadata.accession_number,
                document_type=metadata.form,
                filing_date=metadata.filing_date,
                report_date=metadata.report_date,
                fiscal_period=metadata.fiscal_period,
                source_url=metadata.document_url,
                sections=normalized_sections,
                sector=metadata.sector,
                geography=metadata.geography,
            )
        except Exception as err:
            print(f"Error processing filing {file_path}: {err}")
            return None

    def run(
        self,
        companies: Optional[List[CompanyConfig]] = None,
        force_download: bool = False,
    ) -> List[Document]:
        """Runs full ingestion for all target companies."""
        target_companies = companies or self.config.companies
        all_filings: List[FilingMetadata] = []

        print(f"Discovering filings for {len(target_companies)} companies (2023-2025)...")
        for comp in target_companies:
            filings = self.discover_company_filings(comp)
            all_filings.extend(filings)
            print(f"  [{comp.ticker}] Found {len(filings)} target filings.")

        print(f"\nTotal filings to process: {len(all_filings)}")
        documents: List[Document] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task("Ingesting filings...", total=len(all_filings))

            for filing in all_filings:
                file_path = self.downloader.download_filing(filing, force=force_download)
                if file_path and file_path.exists():
                    doc = self.process_filing_file(file_path, filing)
                    if doc:
                        documents.append(doc)
                progress.advance(task)

        # Save processed documents manifest
        output_dir = Path("data/processed")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "documents.jsonl"
        with open(manifest_path, "w", encoding="utf-8") as f:
            for doc in documents:
                f.write(json.dumps(doc.model_dump()) + "\n")

        print(f"\nSuccessfully ingested {len(documents)} documents. Saved to {manifest_path}")
        return documents
