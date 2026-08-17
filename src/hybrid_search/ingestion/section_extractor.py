"""Extracts targeted high-value narrative and financial sections from SEC filings."""

from __future__ import annotations

import re
from typing import List, Tuple, Dict, Any
from hybrid_search.models.document import DocumentSection


class SectionExtractor:
    """Extracts narrative and financial sections from parsed SEC filing text, filtering Table of Contents traps."""

    PATTERNS_10K = [
        ("ITEM_1", "Item 1. Business", r"(?:^|\n)\s*ITEM\s+1\b[\.\:\s\–\-]+(?:Business|Overview)"),
        ("ITEM_1A", "Item 1A. Risk Factors", r"(?:^|\n)\s*ITEM\s+1A\b[\.\:\s\–\-]+Risk\s+Factors"),
        ("ITEM_1B", "Item 1B. Unresolved Staff Comments", r"(?:^|\n)\s*ITEM\s+1B\b[\.\:\s\–\-]+Unresolved"),
        ("ITEM_1C", "Item 1C. Cybersecurity", r"(?:^|\n)\s*ITEM\s+1C\b[\.\:\s\–\-]+Cybersecurity"),
        ("ITEM_2", "Item 2. Properties", r"(?:^|\n)\s*ITEM\s+2\b[\.\:\s\–\-]+Properties"),
        ("ITEM_3", "Item 3. Legal Proceedings", r"(?:^|\n)\s*ITEM\s+3\b[\.\:\s\–\-]+Legal\s+Proceedings"),
        ("ITEM_5", "Item 5. Market for Common Equity", r"(?:^|\n)\s*ITEM\s+5\b[\.\:\s\–\-]+Market\s+for"),
        ("ITEM_7", "Item 7. Management's Discussion and Analysis", r"(?:^|\n)\s*ITEM\s+7\b[\.\:\s\–\-]+Management(?:'s|\s+Discussion)"),
        ("ITEM_7A", "Item 7A. Quantitative and Qualitative Disclosures", r"(?:^|\n)\s*ITEM\s+7A\b[\.\:\s\–\-]+Quantitative"),
        ("ITEM_8", "Item 8. Financial Statements and Supplementary Data", r"(?:^|\n)\s*ITEM\s+8\b[\.\:\s\–\-]+Financial\s+Statements"),
        ("ITEM_9", "Item 9. Changes in and Disagreements", r"(?:^|\n)\s*ITEM\s+9\b[\.\:\s\–\-]+Changes\s+in"),
        ("ITEM_9A", "Item 9A. Controls and Procedures", r"(?:^|\n)\s*ITEM\s+9A\b[\.\:\s\–\-]+Controls"),
        ("ITEM_10", "Item 10. Directors and Executive Officers", r"(?:^|\n)\s*ITEM\s+10\b[\.\:\s\–\-]+Directors"),
        ("ITEM_11", "Item 11. Executive Compensation", r"(?:^|\n)\s*ITEM\s+11\b[\.\:\s\–\-]+Executive\s+Compensation"),
        ("ITEM_12", "Item 12. Security Ownership of Management", r"(?:^|\n)\s*ITEM\s+12\b[\.\:\s\–\-]+Security\s+Ownership"),
        ("ITEM_13", "Item 13. Certain Relationships and Related Transactions", r"(?:^|\n)\s*ITEM\s+13\b[\.\:\s\–\-]+Certain\s+Relationships"),
        ("ITEM_15", "Item 15. Exhibits and Financial Statement Schedules", r"(?:^|\n)\s*ITEM\s+15\b[\.\:\s\–\-]+Exhibits"),
    ]

    PATTERNS_10Q = [
        ("ITEM_1", "Part I - Item 1. Financial Statements", r"(?:^|\n)\s*(?:PART\s+I[\s\.\–\-]+)?ITEM\s+1\b[\.\:\s\–\-]+Financial\s+Statements"),
        ("ITEM_2", "Part I - Item 2. Management's Discussion and Analysis", r"(?:^|\n)\s*(?:PART\s+I[\s\.\–\-]+)?ITEM\s+2\b[\.\:\s\–\-]+Management(?:'s|\s+Discussion)"),
        ("ITEM_3", "Part I - Item 3. Quantitative and Qualitative Disclosures", r"(?:^|\n)\s*(?:PART\s+I[\s\.\–\-]+)?ITEM\s+3\b[\.\:\s\–\-]+Quantitative"),
        ("ITEM_4", "Part I - Item 4. Controls and Procedures", r"(?:^|\n)\s*(?:PART\s+I[\s\.\–\-]+)?ITEM\s+4\b[\.\:\s\–\-]+Controls"),
        ("ITEM_1A", "Part II - Item 1A. Risk Factors", r"(?:^|\n)\s*(?:PART\s+II[\s\.\–\-]+)?ITEM\s+1A\b[\.\:\s\–\-]+Risk\s+Factors"),
        ("ITEM_2_II", "Part II - Item 2. Unregistered Sales of Equity Securities", r"(?:^|\n)\s*(?:PART\s+II[\s\.\–\-]+)?ITEM\s+2\b[\.\:\s\–\-]+Unregistered"),
        ("ITEM_5", "Part II - Item 5. Other Information", r"(?:^|\n)\s*(?:PART\s+II[\s\.\–\-]+)?ITEM\s+5\b[\.\:\s\–\-]+Other\s+Information"),
        ("ITEM_6", "Part II - Item 6. Exhibits", r"(?:^|\n)\s*(?:PART\s+II[\s\.\–\-]+)?ITEM\s+6\b[\.\:\s\–\-]+Exhibits"),
    ]

    PATTERNS_20F = [
        ("ITEM_3", "Item 3. Key Information", r"(?:^|\n)\s*ITEM\s+3\b[\.\:\s\–\-]+Key\s+Information"),
        ("ITEM_3D", "Item 3.D. Risk Factors", r"(?:^|\n)\s*ITEM\s+3\s*[\.\:\s\–\-]*D\b[\.\:\s\–\-]+Risk\s+Factors"),
        ("ITEM_4", "Item 4. Information on the Company", r"(?:^|\n)\s*ITEM\s+4\b[\.\:\s\–\-]+Information\s+on\s+the\s+Company"),
        ("ITEM_5", "Item 5. Operating and Financial Review and Prospects", r"(?:^|\n)\s*ITEM\s+5\b[\.\:\s\–\-]+Operating\s+and\s+Financial\s+Review"),
        ("ITEM_8", "Item 8. Financial Information", r"(?:^|\n)\s*ITEM\s+8\b[\.\:\s\–\-]+Financial\s+Information"),
        ("ITEM_11", "Item 11. Quantitative and Qualitative Disclosures About Market Risk", r"(?:^|\n)\s*ITEM\s+11\b[\.\:\s\–\-]+Quantitative"),
        ("ITEM_18", "Item 18. Financial Statements", r"(?:^|\n)\s*ITEM\s+18\b[\.\:\s\–\-]+Financial\s+Statements"),
        ("ITEM_19", "Item 19. Exhibits", r"(?:^|\n)\s*ITEM\s+19\b[\.\:\s\–\-]+Exhibits"),
    ]

    @classmethod
    def find_section_spans(cls, text: str, patterns: List[Tuple[str, str, str]]) -> List[Dict[str, Any]]:
        """Finds start positions of actual body sections while ignoring Table of Contents."""
        all_matches: List[Dict[str, Any]] = []

        for code, name, pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                all_matches.append({
                    "code": code,
                    "name": name,
                    "pos": match.start(),
                })

        if not all_matches:
            return []

        all_matches.sort(key=lambda x: x["pos"])

        # Filter out Table of Contents:
        # In a TOC / Index table, multiple item headings appear within close proximity (< 100 chars)
        # or contain dot leaders. Body section headers are spaced hundreds or thousands of characters apart.
        body_spans: List[Dict[str, Any]] = []
        for i, m in enumerate(all_matches):
            next_m = all_matches[i + 1] if i + 1 < len(all_matches) else None
            # If the next section header is within 100 characters, it is a TOC index row
            if next_m and (next_m["pos"] - m["pos"] < 100):
                continue
            body_spans.append(m)

        # De-duplicate: If the same code was matched multiple times in body, keep occurrences
        # or clean spans where length is meaningful (> 50 characters)
        return body_spans

    @classmethod
    def extract_sections(cls, text: str, form_type: str) -> List[DocumentSection]:
        """Extracts targeted narrative and financial sections from filing text."""
        form = form_type.upper().strip()
        if "10-K" in form:
            patterns = cls.PATTERNS_10K
        elif "10-Q" in form:
            patterns = cls.PATTERNS_10Q
        elif "20-F" in form or "6-K" in form:
            patterns = cls.PATTERNS_20F
        else:
            patterns = cls.PATTERNS_10K

        spans = cls.find_section_spans(text, patterns)

        if not spans:
            return [
                DocumentSection(
                    section_name="Full Filing Narrative",
                    section_code="FULL_NARRATIVE",
                    text=text,
                )
            ]

        sections: List[DocumentSection] = []
        for i, span in enumerate(spans):
            code = span["code"]
            name = span["name"]
            start_pos = span["pos"]
            end_pos = spans[i + 1]["pos"] if i + 1 < len(spans) else len(text)

            section_text = text[start_pos:end_pos].strip()

            # Ignore tiny residual snippets (< 100 chars)
            if len(section_text) >= 100:
                sections.append(
                    DocumentSection(
                        section_name=name,
                        section_code=code,
                        text=section_text,
                    )
                )

        if not sections:
            sections.append(
                DocumentSection(
                    section_name="Full Filing Narrative",
                    section_code="FULL_NARRATIVE",
                    text=text,
                )
            )

        return sections
