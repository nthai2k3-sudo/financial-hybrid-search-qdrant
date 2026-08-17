"""Rate-limited, cached SEC filing document downloader."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional
import httpx
from hybrid_search.ingestion.edgar_client import FilingMetadata


class EdgarDownloader:
    """Downloads SEC filings with strict rate limiting, retries, and local caching."""

    def __init__(
        self,
        user_agent: str = "FinancialIntelligenceResearch/1.0 (research@example.com)",
        rate_limit_per_second: float = 8.0,
        request_timeout_seconds: float = 30.0,
        max_retries: int = 3,
        cache_dir: str | Path = "data/raw/edgar",
    ):
        self.user_agent = user_agent
        self.min_interval = 1.0 / max(rate_limit_per_second, 1.0)
        self.request_timeout = request_timeout_seconds
        self.max_retries = max_retries
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_time = 0.0

    def _throttle(self) -> None:
        """Enforces minimum interval between consecutive requests to comply with SEC rules."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_request_time = time.time()

    def get_filing_cache_path(self, filing: FilingMetadata) -> Path:
        """Constructs deterministic local file cache path."""
        cik = filing.cik.zfill(10)
        acc = filing.accession_number.replace("-", "")
        # Clean filename
        filename = filing.primary_document or f"{filing.form.lower()}_{filing.filing_date}.htm"
        return self.cache_dir / cik / acc / filename

    def fetch_json(self, url: str) -> Optional[dict]:
        """Fetches JSON payload with rate limiting and exponential backoff."""
        headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                with httpx.Client(headers=headers, timeout=self.request_timeout) as client:
                    resp = client.get(url)
                    if resp.status_code == 200:
                        return resp.json()
                    elif resp.status_code in (429, 503):
                        sleep_time = (2 ** attempt) * 1.0
                        time.sleep(sleep_time)
                    else:
                        resp.raise_for_status()
            except Exception as err:
                if attempt == self.max_retries - 1:
                    print(f"Failed to fetch {url}: {err}")
                    return None
                time.sleep(1.0 * (attempt + 1))
        return None

    def download_filing(self, filing: FilingMetadata, force: bool = False) -> Optional[Path]:
        """Downloads filing HTML document to local cache if not already present."""
        target_path = self.get_filing_cache_path(filing)
        if target_path.exists() and not force:
            return target_path

        target_path.parent.mkdir(parents=True, exist_ok=True)
        headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"}

        for attempt in range(self.max_retries):
            self._throttle()
            try:
                with httpx.Client(headers=headers, timeout=self.request_timeout) as client:
                    resp = client.get(filing.document_url)
                    if resp.status_code == 200:
                        target_path.write_bytes(resp.content)
                        # Write metadata sidecar JSON
                        meta_path = target_path.parent / "metadata.json"
                        meta_path.write_text(json.dumps(filing.model_dump(), indent=2), encoding="utf-8")
                        return target_path
                    elif resp.status_code in (429, 503):
                        time.sleep((2 ** attempt) * 1.5)
                    else:
                        resp.raise_for_status()
            except Exception as err:
                if attempt == self.max_retries - 1:
                    print(f"Error downloading {filing.document_url}: {err}")
                    return None
                time.sleep(1.0 * (attempt + 1))
        return None
