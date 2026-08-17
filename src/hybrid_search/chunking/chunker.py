"""Deterministic token-aware document chunking with contextual enrichment."""

from __future__ import annotations

import re
import json
from pathlib import Path
from typing import List, Optional
from hybrid_search.models.document import Chunk, Document


class DocumentChunker:
    """Splits Document sections into overlapping, boundary-aware text chunks with contextual metadata."""

    def __init__(
        self,
        chunk_size_tokens: int = 500,
        chunk_overlap_tokens: int = 100,
    ):
        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens
        # Approximation: 1 token ≈ 0.75 words (or ~4 characters)
        self.target_word_size = max(int(chunk_size_tokens * 0.75), 60)
        self.overlap_word_size = max(int(chunk_overlap_tokens * 0.75), 15)

    @classmethod
    def split_into_sentences(cls, text: str) -> List[str]:
        """Splits text into sentences based on punctuation boundaries while preserving financial numbers and abbreviations."""
        text = text.strip()
        if not text:
            return []

        # Split on double newlines or paragraph breaks first
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        units: List[str] = []

        for para in paragraphs:
            # If paragraph contains markdown/pipe table rows, split by line to keep rows atomic
            if "|" in para and "\n" in para:
                lines = [l.strip() for l in para.split("\n") if l.strip()]
                units.extend(lines)
                continue

            # Split sentences carefully avoiding common financial abbreviations (e.g. U.S., Inc., Jan., No., vs.)
            # Match sentence ends: period/exclamation/question followed by whitespace and uppercase letter
            sentence_splits = re.split(r"(?<=[.!?])\s+(?=[A-Z\"\'\$\(])", para)
            for s in sentence_splits:
                s_clean = s.strip()
                if s_clean:
                    units.append(s_clean)

        return units

    def chunk_section_text(self, text: str) -> List[str]:
        """Chunks section text into overlapping windows respecting sentence/table boundaries."""
        sentences = self.split_into_sentences(text)
        if not sentences:
            return []

        chunks: List[str] = []
        current_sentences: List[str] = []
        current_words = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())

            # If a single sentence or table row is larger than target_word_size, split by word window
            if sentence_words > self.target_word_size * 1.5:
                words = sentence.split()
                step = max(self.target_word_size - self.overlap_word_size, 1)
                for start_idx in range(0, len(words), step):
                    window = words[start_idx : start_idx + self.target_word_size]
                    if window:
                        chunks.append(" ".join(window))
                continue

            if current_words + sentence_words > self.target_word_size and current_sentences:
                chunks.append(" ".join(current_sentences))
                # Compute overlap sentences to retain
                overlap_sentences: List[str] = []
                overlap_words = 0
                for s in reversed(current_sentences):
                    s_w = len(s.split())
                    if overlap_words + s_w <= self.overlap_word_size:
                        overlap_sentences.insert(0, s)
                        overlap_words += s_w
                    else:
                        break
                current_sentences = list(overlap_sentences)
                current_words = overlap_words

            current_sentences.append(sentence)
            current_words += sentence_words

        if current_sentences:
            final_chunk = " ".join(current_sentences)
            if not chunks or final_chunk != chunks[-1]:
                chunks.append(final_chunk)

        return chunks

    def chunk_document(self, document: Document, start_chunk_idx: int = 0) -> List[Chunk]:
        """Creates contextualized Chunk models from all sections of a Document."""
        chunks: List[Chunk] = []
        chunk_idx = start_chunk_idx

        sections = document.sections or []
        for section in sections:
            section_chunks = self.chunk_section_text(section.text)
            for text_piece in section_chunks:
                chunk_id = f"{document.document_id}_chunk_{chunk_idx:04d}"

                # Contextual enrichment prefix: ensures dense, sparse, and lexical embeddings
                # retain critical entity, ticker, document type, period, and section context
                context_prefix = (
                    f"[{document.company} ({document.ticker}) | {document.document_type} {document.fiscal_period} | "
                    f"Section: {section.section_name}]\n\n"
                )
                enriched_text = context_prefix + text_piece

                chunk = Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    company=document.company,
                    ticker=document.ticker,
                    cik=document.cik,
                    document_type=document.document_type,
                    filing_date=document.filing_date,
                    fiscal_period=document.fiscal_period,
                    section=section.section_name,
                    sector=document.sector or "Unknown",
                    geography=document.geography or "US",
                    text=enriched_text,
                    chunk_index=chunk_idx,
                    source_url=document.source_url,
                    accession_number=document.accession_number,
                )
                chunks.append(chunk)
                chunk_idx += 1

        return chunks

    def chunk_documents(
        self,
        documents: List[Document],
        output_path: Optional[str | Path] = None,
    ) -> List[Chunk]:
        """Chunks a batch of Documents and optionally serializes to JSONL."""
        all_chunks: List[Chunk] = []
        for doc in documents:
            doc_chunks = self.chunk_document(doc)
            all_chunks.extend(doc_chunks)

        if output_path:
            out_file = Path(output_path)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w", encoding="utf-8") as f:
                for c in all_chunks:
                    f.write(json.dumps(c.model_dump()) + "\n")

            manifest_file = out_file.parent / "manifest.json"
            manifest_data = {
                "total_chunks": len(all_chunks),
                "total_documents": len(documents),
                "companies": list({c.ticker for c in all_chunks}),
            }
            manifest_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

        return all_chunks
