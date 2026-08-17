"""Unit tests for SectionExtractor."""

from hybrid_search.ingestion.section_extractor import SectionExtractor


def test_extract_sections_10k():
    sample_text = """
    Table of Contents
    Item 1. Business ........................... 4
    Item 1A. Risk Factors ...................... 12
    Item 7. Management's Discussion ............ 30

    PART I

    Item 1. Business
    NVIDIA designs graphics processing units and accelerated computing architectures for artificial intelligence, data centers, gaming, and robotics.

    Item 1A. Risk Factors
    We face intense competition in the semiconductor market. Our suppliers, such as TSMC, have limited manufacturing capacity for advanced packaging and H200 accelerators.

    Item 7. Management's Discussion and Analysis of Financial Condition
    During fiscal 2024, our Data Center revenue increased by 217% driven by the NVIDIA HGX platform and Hopper architecture demand.

    Item 8. Financial Statements
    Consolidated Balance Sheets...
    """

    sections = SectionExtractor.extract_sections(sample_text, "10-K")
    assert len(sections) == 3

    codes = [s.section_code for s in sections]
    assert "ITEM_1" in codes
    assert "ITEM_1A" in codes
    assert "ITEM_7" in codes

    risk_sec = next(s for s in sections if s.section_code == "ITEM_1A")
    assert "limited manufacturing capacity" in risk_sec.text
    assert "H200 accelerators" in risk_sec.text


def test_extract_sections_fallback():
    sample_text = "A simple filing document without standard SEC headers describing corporate events and operations."
    sections = SectionExtractor.extract_sections(sample_text, "10-K")
    assert len(sections) >= 1
    assert "describing corporate events" in sections[0].text
