# ADR-007: Evaluation Methodology, Golden Dataset Design & Metrics

## Status
Accepted

## Context & Problem Statement
A common failure mode in retrieval evaluation is using a monolithic benchmark metric (e.g. overall NDCG on a single homogeneous query distribution) and concluding that one method is globally superior. In reality, retrieval systems exhibit trade-offs:
- Exact term search (BM25) excels at unique SKUs and fails at conceptual paraphrasing.
- Dense retrieval excels at semantic concepts and fails at exact alphanumeric identifiers.
- Hybrid retrieval resolves conflicts but introduces additional index and compute overhead.

To demonstrate where each configuration works best with empirical evidence from financial data, the evaluation framework must categorize queries by linguistic and structural intent and measure performance across multiple complementary metrics.

## Evaluation Dataset Taxonomy (5 Query Categories)

The evaluation suite ([`evaluation/golden_dataset.yaml`](../../evaluation/golden_dataset.yaml)) defines 25 hand-curated financial queries across 5 distinct categories:

| Category | Query Count | Linguistic & Search Characteristics | Expected Strongest Configurations |
|---|:---:|---|---|
| **1. Exact Terminology** | 5 | Exact product SKUs (`H200`), ticker symbols, specific legal accounting rules (`ASC 606`). | **BM25 / Sparse** (Dense struggles with rare tokens) |
| **2. Semantic Questions** | 5 | Conceptual questions with significant vocabulary mismatch (*"capex reduction"* vs *"slowing infrastructure investment"*). | **Dense** (BM25 fails due to term mismatch) |
| **3. Mixed Queries** | 5 | Named corporate entities paired with conceptual strategic themes (*"Microsoft Azure capital expenditure for OpenAI"*). | **BM25 + Dense / Sparse + Dense** (Requires both signals) |
| **4. Rare Terminology** | 5 | Low-frequency domain jargon, specialized packaging acronyms (`CoWoS`, `VIE`, `RPO`). | **BM25 / Sparse** (Lexical precision wins) |
| **5. Long Natural Language** | 5 | Multi-clause analyst-style research inquiries. | **Sparse + Dense / Dense** (Captures complex intent) |

## Evaluation Metrics Formulations

1. **Recall@K (Coverage):**
   $$\text{Recall@K} = \frac{|\text{Relevant} \cap \text{Retrieved@K}|}{|\text{Relevant}|}$$

2. **Precision@K (Density):**
   $$\text{Precision@K} = \frac{|\text{Relevant} \cap \text{Retrieved@K}|}{K}$$

3. **Mean Reciprocal Rank (MRR - Top Rank Position):**
   $$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_{\text{first\_hit}}(q)}$$

4. **Normalized Discounted Cumulative Gain (NDCG@K - Graded Quality):**
   $$\text{DCG@K} = \sum_{i=1}^K \frac{rel_i}{\log_2(i + 1)}, \quad \text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

5. **Hit Rate@K:**
   $$\text{HitRate@K} = \mathbb{I}(|\text{Relevant} \cap \text{Retrieved@K}| > 0)$$

6. **Multi-Phase Latency Profiling:**
   - Query Encoding Time (ONNX CPU execution)
   - Qdrant Vector/Sparse Retrieval & Fusion Time
   - Total End-to-End Latency ($p_{50}, p_{95}, p_{99}$, Mean, StdDev)

## Limitations & Annotator Note
Relevance judgments are curated by domain-expert financial text analysis. The dataset represents a focused benchmark designed for verifiable local reproducibility (~25 queries, ~50 filings) rather than a multi-million document industrial TREC corpus.
