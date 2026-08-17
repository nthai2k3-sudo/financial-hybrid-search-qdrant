"""Neural & cross-feature reranking module for precision financial retrieval."""

from __future__ import annotations

import re
from typing import List, Optional
from hybrid_search.models.search import SearchResult
from hybrid_search.retrieval.extractor import AnswerExtractor


class NeuralReranker:
    """Reranks candidate search results using cross-feature query alignment and extractive precision."""

    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """Reranks a list of SearchResults using joint query-document cross-scoring."""
        if not results:
            return []

        query_tokens = [w.lower() for w in re.findall(r"\w+", query)]
        key_tokens = set(
            w for w in query_tokens
            if w not in AnswerExtractor.STOP_WORDS and len(w) > 1
        )
        generic_words = {"fiscal", "year", "revenue", "sales", "total", "annual", "quarter", "income", "segment", "net"}
        entity_tokens = set(w for w in key_tokens if w not in generic_words)

        # Automatic Query Intent Classification (Annual vs Quarterly)
        q_lower = query.lower()
        annual_keywords = ["fiscal year", "annual", "full year", "whole year", "for the year", "10-k", "20-f"]
        is_annual_query = any(k in q_lower for k in annual_keywords) or bool(re.search(r"\bfy\s*202[0-9]\b", q_lower))
        is_quarterly_query = any(k in q_lower for k in ["quarter", "q1", "q2", "q3", "q4", "three months", "nine months", "10-q", "first quarter", "second quarter", "third quarter", "fourth quarter"])
        year_matches = re.findall(r"\b(202[0-9])\b", query)
        target_year = year_matches[0] if year_matches else None

        # Default full-year revenue / metric questions to Annual 10-K if not quarterly
        if not is_quarterly_query and target_year and any(m in q_lower for m in ["revenue", "sales", "net income", "total", "profit", "r&d", "employees", "segment"]):
            is_annual_query = True

        scored_results = []
        for rank_idx, result in enumerate(results):
            # 1. Extract the best answering sentence/row
            extracted = AnswerExtractor.extract(query, result.text)
            result.exact_answer = extracted.exact_answer
            result.highlighted_text = extracted.highlighted_text
            result.answer_confidence = extracted.confidence

            # 2. Compute Cross-Feature Score
            text_tokens = set(re.findall(r"\w+", result.text.lower()))
            ans_tokens = set(re.findall(r"\w+", extracted.exact_answer.lower()))

            # Entity-weighted answer coverage
            if entity_tokens:
                entity_ans_match = len(entity_tokens.intersection(ans_tokens)) / len(entity_tokens)
                entity_text_match = len(entity_tokens.intersection(text_tokens)) / len(entity_tokens)
            else:
                entity_ans_match = 0.0
                entity_text_match = 0.0

            gen_ans_match = len(key_tokens.intersection(ans_tokens)) / (len(key_tokens) + 1e-5)
            gen_text_match = len(key_tokens.intersection(text_tokens)) / (len(key_tokens) + 1e-5)

            ans_coverage = 0.70 * entity_ans_match + 0.30 * gen_ans_match
            coverage = 0.70 * entity_text_match + 0.30 * gen_text_match

            # Numerical/Metric Alignment Bonus
            metric_bonus = 0.0
            if any(q in q_lower for q in ["how many", "percentage", "%", "revenue", "percent", "top", "500", "sales", "income"]):
                if re.search(r"\d+[\.,]?\d*%", extracted.exact_answer) or re.search(r"\b\d{1,3}(?:,\d{3})*\b", extracted.exact_answer) or "$" in extracted.exact_answer:
                    metric_bonus += 0.25

            # Table Fact Bonus - strong boost if exact aligned row was extracted
            table_bonus = 0.35 if (extracted.is_table_row and "->" in extracted.exact_answer) else (0.15 if extracted.is_table_row else 0.0)

            # Intent & Granularity Scoring
            intent_score = 0.0
            if is_annual_query:
                if result.document_type in ["10-K", "20-F"]:
                    intent_score += 0.30
                    if target_year and target_year in result.fiscal_period:
                        intent_score += 0.35
                    if target_year and target_year in extracted.exact_answer:
                        intent_score += 0.15
                elif result.document_type == "10-Q":
                    intent_score -= 0.45  # Decisive penalty for quarterly filing answering full-year query
            elif is_quarterly_query:
                if result.document_type == "10-Q":
                    intent_score += 0.30
                    if target_year and target_year in result.fiscal_period:
                        intent_score += 0.25
                elif result.document_type in ["10-K", "20-F"]:
                    intent_score -= 0.35

            # Original retrieval score normalized
            base_score = float(result.score)
            if base_score > 1.0:
                base_score = base_score / (base_score + 50.0)  # Normalize BM25/SPLADE scores into [0, 1]

            # Composite Precision Score
            cross_score = (
                0.15 * base_score +
                0.25 * ans_coverage +
                0.10 * coverage +
                0.15 * metric_bonus +
                0.15 * table_bonus +
                0.20 * intent_score
            )

            scored_results.append((cross_score, result))




        # Sort descending by composite cross-score
        scored_results.sort(key=lambda x: x[0], reverse=True)

        k = top_k or len(results)
        final_results: List[SearchResult] = []
        for new_rank, (new_score, res) in enumerate(scored_results[:k], start=1):
            res.rank = new_rank
            res.score = round(new_score, 4)
            res.retrieval_method = "rerank"
            final_results.append(res)

        return final_results
