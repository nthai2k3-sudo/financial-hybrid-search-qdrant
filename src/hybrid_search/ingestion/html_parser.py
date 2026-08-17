"""HTML and iXBRL parser for SEC filing documents."""

from __future__ import annotations

import re
import html
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# Suppress XMLParsedAsHTMLWarning when BeautifulSoup parses SEC iXBRL/HTML documents
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


class EdgarHtmlParser:
    """Parses SEC filing HTML/iXBRL documents into clean, searchable plain text with preserved tables."""

    @classmethod
    def parse_html_to_text(cls, html_content: str | bytes) -> str:
        """Converts raw filing HTML into clean narrative text preserving tabular structure."""
        if isinstance(html_content, bytes):
            try:
                html_content = html_content.decode("utf-8")
            except UnicodeDecodeError:
                html_content = html_content.decode("latin-1", errors="ignore")

        soup = BeautifulSoup(html_content, "lxml")

        # Strip unneeded non-content tags
        for tag in soup(["script", "style", "meta", "link", "noscript", "ix:header", "ix:hidden"]):
            tag.decompose()

        # Format HTML tables into clean structured text with cell separation
        for td in soup.find_all(["td", "th"]):
            td.append(" | ")
        for tr in soup.find_all("tr"):
            tr.append("\n")

        # Insert newlines after block elements
        for tag in soup.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "br", "li", "section", "article", "header", "footer", "blockquote", "hr"]):
            tag.append("\n")

        # Extract text
        text = soup.get_text()

        # Unescape HTML entities
        text = html.unescape(text)

        # Normalize line endings and non-breaking spaces
        text = text.replace("\xa0", " ").replace("\u200b", "").replace("\ufeff", "")
        text = re.sub(r"\r\n|\r", "\n", text)

        # Clean multiple spaces and tabs (preserving newlines)
        text = re.sub(r"[ \t]+", " ", text)

        # Clean excessive blank lines (more than 2 consecutive newlines)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Trim each line
        lines = [line.strip() for line in text.split("\n")]
        cleaned_text = "\n".join(lines).strip()
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

        return cleaned_text
