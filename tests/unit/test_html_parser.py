"""Unit tests for EdgarHtmlParser."""

from hybrid_search.ingestion.html_parser import EdgarHtmlParser


def test_parse_html_basic():
    sample_html = """
    <html>
        <head>
            <title>NVIDIA 10-K</title>
            <style>body { font-size: 12px; }</style>
            <script>console.log('tracker');</script>
        </head>
        <body>
            <ix:header><ix:hidden>XBRL data</ix:hidden></ix:header>
            <h1>Item 1. Business</h1>
            <p>NVIDIA pioneered <b>GPU-accelerated</b> computing.</p>
            <p>Our revenue grew by &gt; 100% in FY24.&nbsp;Demand remains strong.</p>
        </body>
    </html>
    """
    text = EdgarHtmlParser.parse_html_to_text(sample_html)
    assert "Item 1. Business" in text
    assert "NVIDIA pioneered GPU-accelerated computing." in text
    assert "Our revenue grew by > 100% in FY24. Demand remains strong." in text
    assert "console.log" not in text
    assert "XBRL data" not in text
    assert "style" not in text
