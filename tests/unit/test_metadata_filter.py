"""Unit tests for QdrantFilterBuilder."""

from hybrid_search.filtering.metadata_filter import QdrantFilterBuilder
from hybrid_search.models.filter import MetadataFilter


def test_build_filter_empty():
    f = MetadataFilter()
    q_filter = QdrantFilterBuilder.build_filter(f)
    assert q_filter is None


def test_build_filter_single_ticker():
    f = MetadataFilter(tickers=["NVDA"])
    q_filter = QdrantFilterBuilder.build_filter(f)
    assert q_filter is not None
    assert len(q_filter.must) == 1
    cond = q_filter.must[0]
    assert cond.key == "ticker"
    assert cond.match.value == "NVDA"


def test_build_filter_multi_ticker_and_date():
    f = MetadataFilter(
        tickers=["NVDA", "AAPL"],
        document_types=["10-K"],
        filing_date_start="2024-01-01",
        filing_date_end="2024-12-31",
    )
    q_filter = QdrantFilterBuilder.build_filter(f)
    assert q_filter is not None
    assert len(q_filter.must) == 3
