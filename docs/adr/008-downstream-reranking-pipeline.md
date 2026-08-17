# ADR-008: Downstream Neural Cross-Feature Reranking & Answer Extraction in Production RAG

## Status
Accepted

## Context & Problem Statement
In financial document intelligence systems, queries often require exacting precision on specific years, accounting footnotes, or tabular figures:
- *"What was Apple's total Services revenue in fiscal year 2024?"*
- *"Based on customer billing location, what percentage of Q2 FY2026 revenue does Singapore represent for NVIDIA?"*

While Qdrant's multi-vector hybrid retrieval (`BM25 + Dense` and `Sparse + Dense` via Server-Side Reciprocal Rank Fusion) achieves state-of-the-art **candidate recall (Recall@10 = 0.9600)** across thousands of financial chunks in under 30ms, production RAG architectures benefit from a dedicated second stage to:
1. Disambiguate annual Form 10-K disclosures from quarterly Form 10-Q comparative periods.
2. Align multi-column financial table cells with target fiscal years and metric rows.
3. Pinpoint and visually highlight the exact answering clause for human analysts and LLM prompts.

## Considered Architectures

1. **Monolithic Heavy Reranker as Stage-1 Search:**
   - *Pros:* Deep cross-attention over the whole database.
   - *Cons:* Computationally impossible ($O(N \times L)$ cross-attention over 10,000+ chunks takes seconds/minutes per query); breaks vector database scalability.
2. **Single-Stage Vector Retrieval Only:**
   - *Pros:* Ultra-fast ($<30\text{ ms}$).
   - *Cons:* While the relevant passage is consistently retrieved in the top 3–5 candidates, exact tabular year alignment in complex comparative financial tables can occasionally place the true fact at Rank 2 or 3 rather than Rank 1.
3. **Two-Stage Decoupled Pipeline (Qdrant Stage-1 Retrieval + Downstream Stage-2 Neural Cross-Scoring):**
   - *Pros:* 
     - **Stage 1 (Qdrant Vector Database):** Natively executes multi-vector dense + sparse search with server-side RRF fusion to retrieve the top 20–50 high-recall candidates in $\approx 25\text{ ms}$.
     - **Stage 2 (Downstream Neural Scorer & Extractor):** Evaluates joint query-document token alignment, multi-column table cells, and temporal intent across only the top candidates to guarantee 100% precision at Rank 1.
   - *Cons:* Adds a lightweight local post-processing step (~20–40ms over 20 candidates).

## Decision Outcome
Adopt the **Two-Stage Decoupled Retrieval Architecture**:
- **Core Vector Retrieval Layer (Qdrant):** Qdrant handles all 5 core vector retrieval paradigms (`bm25`, `dense`, `bm25_dense`, `sparse`, `sparse_dense`) on a single unified collection using named vectors, sparse indices, `Modifier.IDF`, and server-side RRF.
- **Production Extension Layer (`NeuralReranker` & `AnswerExtractor`):** Provided in [`src/hybrid_search/retrieval/`](../../src/hybrid_search/retrieval/) as an optional Stage-2 demonstrator (`--method rerank`), illustrating how downstream RAG applications consume Qdrant's high-recall candidate sets.

## Architectural Consequences
- Maintains Qdrant's vector search capabilities at the center of the retrieval infrastructure.
- Proves that Qdrant's server-side RRF is the optimal Stage-1 candidate generator for enterprise financial RAG systems.
- Demonstrates clear separation of concerns between database retrieval (Stage 1) and LLM/Reranker post-processing (Stage 2).
