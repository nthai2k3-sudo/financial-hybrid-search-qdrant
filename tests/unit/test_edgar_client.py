"""Unit tests for EdgarClient submissions parser."""

from hybrid_search.ingestion.edgar_client import EdgarClient


def test_format_cik():
    assert EdgarClient.format_cik("1045810") == "0001045810"
    assert EdgarClient.format_cik(320193) == "0000320193"


def test_build_document_url():
    url = EdgarClient.build_document_url(
        cik="0001045810",
        accession_number="0001045810-24-000029",
        primary_document="nvda-20240128.htm",
    )
    assert url == "https://www.sec.gov/Archives/edgar/data/1045810/000104581024000029/nvda-20240128.htm"


def test_parse_submissions_json():
    mock_submissions = {
        "cik": "1045810",
        "filings": {
            "recent": {
                "form": ["10-K", "10-Q", "10-Q", "10-Q", "8-K"],
                "filingDate": ["2024-02-21", "2024-05-22", "2024-08-28", "2024-11-20", "2024-01-10"],
                "reportDate": ["2024-01-28", "2024-04-28", "2024-07-28", "2024-10-27", "2024-01-10"],
                "accessionNumber": [
                    "0001045810-24-000029",
                    "0001045810-24-000062",
                    "0001045810-24-000088",
                    "0001045810-24-000115",
                    "0001045810-24-000005",
                ],
                "primaryDocument": [
                    "nvda-20240128.htm",
                    "nvda-20240428.htm",
                    "nvda-20240728.htm",
                    "nvda-20241027.htm",
                    "nvda-8k.htm",
                ],
            }
        },
    }

    filings = EdgarClient.parse_submissions_json(
        submissions=mock_submissions,
        ticker="NVDA",
        company_name="NVIDIA Corporation",
        target_forms=["10-K", "10-Q"],
        start_year=2023,
        end_year=2025,
        max_10q_per_company=2,
    )

    # 1 10-K and capped at 2 10-Qs = 3 total filings
    assert len(filings) == 3
    assert filings[0].form == "10-K"
    assert filings[0].ticker == "NVDA"
    assert filings[1].form == "10-Q"
    assert filings[2].form == "10-Q"
