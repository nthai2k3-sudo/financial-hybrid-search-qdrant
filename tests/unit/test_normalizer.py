"""Unit tests for TextNormalizer."""

from hybrid_search.ingestion.normalizer import TextNormalizer


def test_normalizer_quotes_and_dashes():
    raw = "NVIDIA\u2019s \u201cH200\u201d GPU \u2014 accelerated computing \u2013 high performance."
    cleaned = TextNormalizer.normalize(raw)
    assert cleaned == 'NVIDIA\'s "H200" GPU - accelerated computing - high performance.'


def test_normalizer_financial_terms_preserved():
    raw = "CapEx for FY2024 was $15.2B under ASC 606 with Non-GAAP EPS growth."
    cleaned = TextNormalizer.normalize(raw)
    assert "$15.2B" in cleaned
    assert "ASC 606" in cleaned
    assert "Non-GAAP EPS" in cleaned
    assert "CapEx" in cleaned


def test_normalizer_hyphenated_linebreaks():
    raw = "The company expanded its cap-\nital expenditure in 2024."
    cleaned = TextNormalizer.normalize(raw)
    assert "capital expenditure" in cleaned
