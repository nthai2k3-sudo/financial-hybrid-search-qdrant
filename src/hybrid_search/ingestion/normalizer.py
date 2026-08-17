"""Financial text normalization utilities."""

from __future__ import annotations

import re
import unicodedata


class TextNormalizer:
    """Normalizes financial text while preserving financial terms, tickers, and acronyms."""

    UNICODE_REPLACEMENTS = {
        "\u201c": '"',  # Left double quote
        "\u201d": '"',  # Right double quote
        "\u2018": "'",  # Left single quote
        "\u2019": "'",  # Right single quote
        "\u2014": " - ",  # Em dash
        "\u2013": " - ",  # En dash
        "\u2212": "-",  # Minus sign
        "\u2022": "* ",  # Bullet
        "\u2026": "...",  # Ellipsis
        "\u00a0": " ",  # Non-breaking space
    }

    @classmethod
    def normalize(cls, text: str) -> str:
        """Applies normalization passes to financial narrative text."""
        if not text:
            return ""

        # Normalize unicode characters
        for char, replacement in cls.UNICODE_REPLACEMENTS.items():
            text = text.replace(char, replacement)

        # NFKC Unicode normalization
        text = unicodedata.normalize("NFKC", text)

        # Normalize line breaks
        text = re.sub(r"\r\n|\r", "\n", text)

        # Remove hyphenation at line breaks (e.g. "cap-\nital" -> "capital")
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

        # Clean multiple spaces and tabs within lines
        text = re.sub(r"[ \t]+", " ", text)

        # Clean excessive consecutive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
