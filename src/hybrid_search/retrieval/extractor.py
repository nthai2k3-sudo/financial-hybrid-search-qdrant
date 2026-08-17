"""Query-aware extractive answer pinpointer and highlighter for financial texts and tables."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class ExtractedAnswer:
    """Encapsulates the pinpointed answer and highlighted text."""
    exact_answer: str
    confidence: float
    highlighted_text: str
    is_table_row: bool = False


class AnswerExtractor:
    """Extracts the precise answering clause, sentence, or table row from a retrieved chunk."""

    STOP_WORDS = {
        "what", "is", "the", "of", "and", "a", "an", "in", "to", "for", "how", "many",
        "based", "on", "does", "represents", "represent", "which", "are", "from", "with",
        "by", "at", "as", "was", "were", "or", "that", "this", "these", "those", "about"
    }

    @classmethod
    def extract(cls, query: str, chunk_text: str) -> ExtractedAnswer:
        """Extracts the most relevant answering sentence or table segment for the given query."""
        if not chunk_text:
            return ExtractedAnswer(exact_answer="", confidence=0.0, highlighted_text="")

        # Separate metadata header from body
        header_match = re.match(r"^(\[.*?\])\s*\n+", chunk_text)
        header = header_match.group(1) if header_match else ""
        body = chunk_text[len(header):].strip() if header else chunk_text.strip()

        # 1. Try smart tabular row extraction if text contains tables
        table_fact = cls._extract_table_fact(query, body)
        if table_fact:
            highlighted_body = cls._highlight_in_text(body, table_fact[1])
            full_highlighted = f"{header}\n\n{highlighted_body}" if header else highlighted_body
            return ExtractedAnswer(
                exact_answer=table_fact[0],
                confidence=0.95,
                highlighted_text=full_highlighted,
                is_table_row=True,
            )

        # 2. Parse candidates: sentences, footnotes, and table rows
        candidates = cls._split_into_candidates(body)
        if not candidates:
            return ExtractedAnswer(exact_answer=body[:200], confidence=0.5, highlighted_text=chunk_text)

        # Tokenize query keywords
        query_words = [w.lower() for w in re.findall(r"\w+", query)]
        keywords = [w for w in query_words if w not in cls.STOP_WORDS and len(w) > 1]
        keyword_set = set(keywords)

        scored_candidates: List[Tuple[float, str, bool]] = []
        for cand, is_table in candidates:
            cand_words = set(re.findall(r"\w+", cand.lower()))
            overlap = len(keyword_set.intersection(cand_words))

            # Bonus for numerical/percentage facts if query asks for percentage/how many
            bonus = 0.0
            if any(q in query.lower() for q in ["how many", "percentage", "revenue", "percent", "%", "total", "powers"]):
                if re.search(r"\d+[\.,]?\d*%", cand) or re.search(r"\b\d{1,3}(?:,\d{3})*\b", cand) or "$" in cand or "%" in cand:
                    bonus += 0.25

            # Priority for specific company / geographic named entity matches
            for kw in keyword_set:
                if kw in cand_words and len(kw) >= 4:
                    bonus += 0.1

            score = (overlap / (len(keyword_set) + 1e-5)) + bonus
            scored_candidates.append((score, cand, is_table))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_cand, is_table = scored_candidates[0]

        # Clean candidate text for display
        clean_answer = cls._clean_answer_text(best_cand)

        # Highlight the answer inside the body
        highlighted_body = cls._highlight_in_text(body, best_cand)
        full_highlighted = f"{header}\n\n{highlighted_body}" if header else highlighted_body

        confidence = min(round(best_score, 4), 1.0)
        return ExtractedAnswer(
            exact_answer=clean_answer,
            confidence=confidence,
            highlighted_text=full_highlighted,
            is_table_row=is_table,
        )

    @classmethod
    def _extract_table_fact(cls, query: str, body: str) -> Optional[Tuple[str, str]]:
        """Extracts aligned tabular row metric (e.g. 'Intelligent Cloud -> 2024: $105,362M') if present."""
        if "|" not in body:
            return None

        # 1. Check for Footnotes first (e.g., Singapore represented 22%...)
        fn_match = re.search(r"(\(\d+\)\s+[A-Z][^\n]+(?:represented|revenue|accounted)[^\n]+)", body, re.IGNORECASE)
        if fn_match:
            fn_text = fn_match.group(1).strip()
            stop_words = {"what", "was", "the", "for", "in", "of", "and", "a", "an", "to", "how", "much", "is"}
            q_words = set(re.findall(r"\w+", query.lower())) - stop_words
            fn_words = set(re.findall(r"\w+", fn_text.lower()))
            if len(q_words.intersection(fn_words)) >= 2:
                sents = re.split(r"(?<=[.!?])\s+", fn_text)
                clean_fn = sents[0] if sents else fn_text
                return (clean_fn, fn_text)

        # 2. Parse Clean Table Cells
        cells = [c.strip() for c in body.split("|") if c.strip() and c.strip() not in ["$", "%", "-"]]
        if len(cells) < 4:
            return None

        # 3. Detect Header Years (e.g. 2024, 2023, 2022)
        years: List[str] = []
        for c in cells[:30]:
            if len(c) < 30 and not any(k in c.lower() for k in ["form", "note", "item"]):
                found = re.findall(r"\b(202[0-9]|201[0-9])\b", c)
                for y in found:
                    if y not in years:
                        years.append(y)

        # Fallback: scan all cells for standard 3-year filing sequence
        if not years:
            for c in cells:
                found_years = re.findall(r"\b(202[0-9]|201[0-9])\b", c)
                for y in found_years:
                    if y not in years:
                        years.append(y)
                if len(years) >= 3:
                    break

        # Fallback 2: infer 3-year sequence from contextual header or query year
        if not years:
            fy_match = re.search(r"\bFY(202[0-9])\b", body) or re.search(r"\b(202[0-9])\b", query)
            if fy_match:
                base_yr = int(fy_match.group(1))
                years = [str(base_yr), str(base_yr - 1), str(base_yr - 2)]

        company_words = {"apple", "inc", "microsoft", "corp", "corporation", "nvidia", "tesla", "amazon", "alphabet", "google", "meta", "intel", "amd", "tsmc"}
        stop_words = {"what", "was", "the", "for", "in", "of", "and", "a", "an", "to", "how", "much", "is", "total", "revenue", "fiscal", "year", "2024", "2023", "2022", "2025", "2026"}.union(company_words)
        q_metric_words = [w for w in re.findall(r"\w+", query.lower()) if w not in stop_words and len(w) > 1]
        if not q_metric_words:
            q_metric_words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2 and w not in company_words]

        # 4. Look for Target Row Labels (skipping dates, headers, company names)
        target_matches = []
        for idx, cell in enumerate(cells):
            if len(cell) > 50:
                continue
            if re.search(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December|Year Ended|Months Ended|Form 10|Performance|Inc\.|Corporation)\b", cell, re.IGNORECASE):
                continue
            if re.match(r"^[\$\d\s,\.\(\)\-\%]+$", cell):
                continue

            cell_lower = cell.lower()
            match_count = sum(1 for qw in q_metric_words if qw in cell_lower)
            if match_count > 0:
                target_matches.append((match_count, idx, cell))

        if not target_matches:
            return None

        target_matches.sort(key=lambda x: x[0], reverse=True)
        best_count, best_idx, best_label = target_matches[0]

        # 5. Extract Multi-Year Numerical Values Following Label
        values: List[str] = []
        for cell in cells[best_idx + 1:]:
            cleaned_cell = cell.replace("$", "").replace(",", "").strip()
            if re.match(r"^\(?\d+(?:\.\d+)?\)?%?$", cleaned_cell):
                if "," in cell or (cleaned_cell.isdigit() and int(cleaned_cell) >= 100):
                    values.append(cell.replace("$", "").strip())
                    if len(values) >= (len(years) if years else 3):
                        break
            elif len(cell) > 3 and not cell.isdigit():
                break

        if values:
            if years and len(years) <= len(values):
                parts = [f"{y}: ${v}M" for y, v in zip(years, values)]
                return (f"{best_label} -> " + " | ".join(parts), best_label)
            elif years and len(values) > 0:
                parts = [f"{years[i]}: ${v}M" if i < len(years) else f"${v}M" for i, v in enumerate(values)]
                return (f"{best_label} -> " + " | ".join(parts), best_label)
            else:
                return (f"{best_label} -> " + " | ".join([f"${v}M" for v in values]), best_label)

        return None



    @classmethod
    def _split_into_candidates(cls, text: str) -> List[Tuple[str, bool]]:
        """Splits text into meaningful semantic candidates (sentences, bullet items, footnotes, table segments)."""
        candidates: List[Tuple[str, bool]] = []

        # 1. Detect Footnotes like (1) Singapore represented... or Footnote 1: ...
        footnote_pattern = re.compile(r"(\(\d+\)\s+[A-Z][^\n]+)")
        for fn in footnote_pattern.findall(text):
            if len(fn.strip()) > 20:
                candidates.append((fn.strip(), False))

        # 2. Split lines for table rows vs standard narrative text
        lines = text.split("\n")
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Clean leading/trailing pipes
            clean_line = re.sub(r"^[\|\s]+", "", line_str).strip()
            if not clean_line:
                continue

            # Check if this line is genuinely a tabular row (multiple short cells) vs narrative text
            cells = [c.strip() for c in clean_line.split("|") if c.strip()]
            is_true_table = (
                len(cells) >= 3 and
                all(len(c) < 70 for c in cells) and
                any(re.search(r"\d", c) for c in cells)
            )

            if is_true_table:
                # Genuine table row
                candidates.append((clean_line, True))
            else:
                # Narrative paragraph (even if it had pipe artifacts): split by bullets (*, •) and sentence boundaries
                # Remove isolated stray pipes from narrative text
                narrative_body = re.sub(r"\|+", " ", clean_line)
                bullet_parts = re.split(r"(?:\s*[\*\•]\s+|\n\s*[-*]\s+)", narrative_body)
                for bp in bullet_parts:
                    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\"])", bp.strip())
                    for s in sentences:
                        s_clean = s.strip()
                        if len(s_clean) > 20:
                            candidates.append((s_clean, False))

        return candidates

    @classmethod
    def _clean_answer_text(cls, text: str) -> str:
        """Cleans extracted candidate string into a concise, readable fact."""
        cleaned = text.strip()
        if "|" in cleaned:
            cells = [c.strip() for c in cleaned.split("|") if c.strip() and c.strip() != "$"]
            cleaned = " | ".join(cells)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    @classmethod
    def _highlight_in_text(cls, body: str, target: str) -> str:
        """Surrounds the target sentence inside the body text with visual highlight tags."""
        if not target:
            return body
        if target in body:
            return body.replace(target, f"[bold green]{target}[/bold green]")
        escaped_target = re.escape(target[:35])
        return re.sub(f"({escaped_target}[^\n.!?]*[.!?]?)", r"[bold green]\1[/bold green]", body, count=1)



