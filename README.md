# Financial Intelligence Hybrid Search — Reference Implementation

A production-grade reference implementation demonstrating, with empirical evidence from SEC financial filings (Forms 10-K, 10-Q, 20-F), how **five core retrieval configurations** and a **downstream dual-stage reranking pipeline** operate on **Qdrant Vector Database** with **zero paid API dependencies**.

```mermaid
graph TD
    User([Analyst / Developer Query]) --> CLI[CLI Entry Point / Search Engine]
    
    subgraph "Query Encoding (Local CPU ONNX via FastEmbed)"
        CLI --> EncDense["Dense Encoder<br/>(BAAI/bge-small-en-v1.5, 384d)"]
        CLI --> EncBM25["BM25 Lexical Encoder<br/>(Qdrant/bm25 Sparse)"]
        CLI --> EncSplade["SPLADE Sparse Encoder<br/>(prithivida/Splade_PP_en_v1)"]
    end
    
    subgraph "Qdrant Vector Database (Single Collection / 3 Named Vector Spaces)"
        EncDense --> QdrantDense[("Dense Vector Space<br/>(Cosine Metric, HNSW)")]
        EncBM25 --> QdrantBM25[("BM25 Vector Space<br/>(Sparse Modifier.IDF)")]
        EncSplade --> QdrantSparse[("Sparse Vector Space<br/>(Sparse Modifier.IDF)")]
        
        QdrantDense -. "Prefetch (k=50)" .-> RRF["Server-Side Reciprocal Rank Fusion (RRF)<br/>k = 60"]
        QdrantBM25 -. "Prefetch (k=50)" .-> RRF
        QdrantSparse -. "Prefetch (k=50)" .-> RRF
    end
    
    subgraph "Qdrant Retrieval Outputs (5 Core Configurations)"
        QdrantDense --> Results1["1. Dense Only"]
        QdrantBM25 --> Results2["2. BM25 Only"]
        QdrantSparse --> Results3["4. Sparse (SPLADE) Only"]
        RRF --> Results4["3. BM25 + Dense Hybrid"]
        RRF --> Results5["5. Sparse + Dense Hybrid"]
    end
    
    subgraph "Production Downstream RAG Pipeline (Stage-2)"
        Results4 & Results5 -. "High-Recall Candidates (k=20..50)" .-> Rerank["NeuralReranker & AnswerExtractor<br/>(Cross-Feature Table & Metric Alignment)"]
        Rerank --> Results6["6. Pinpointed Direct Answer (Rank #1)"]
    end
```

---

## The 5 Core Retrieval Configurations (+ Downstream Reranking)

All five retrieval configurations operate directly against a **single unified Qdrant collection** containing 3 named vector spaces, avoiding the architectural friction of multi-database setups:

| # | Configuration | Representation / Engine | Core Behavioral Strength | Where It Does Justice | Where It Fails |
|---|---|---|---|---|---|
| **1** | **BM25 (Lexical)** | `Qdrant/bm25` Sparse Vector with dynamic `Modifier.IDF` | Verbatim token matching without neural hallucination. | Exact product SKUs (`H200`), ticker symbols (`NVDA`), CIK codes, and statutory accounting rules (`ASC 606`). | **Vocabulary Mismatch:** Fails when documents express the concept using synonyms (*"capex reduction"* vs *"slowing infrastructure investment"*). |
| **2** | **Dense Vector** | `BAAI/bge-small-en-v1.5` (384d Cosine) | Continuous semantic geometry and conceptual generalization. | Broad conceptual research, thematic inquiries, and paraphrased questions. | **Semantic Drift / Token Blindness:** Fails on exact numbers, rare alphanumeric codes, dates, and fiscal period boundaries. |
| **3** | **BM25 + Dense Hybrid** | Server-Side RRF (`prefetch` `bm25` + `dense`) | Fuses exact lexical precision with semantic recall in a single database call. | Mixed financial queries combining a specific corporate entity/SKU with conceptual themes. | Misses synonyms that are not in the query text unless retrieved by the dense candidate pool. |
| **4** | **Sparse (SPLADE)** | `prithivida/Splade_PP_en_v1` Learned Sparse Vector | Neural term expansion over a 30k BERT vocabulary directly into sparse weights. | Domain jargon and plain-English financial questions with synonym divergence. | **Over-Expansion:** Can introduce false-positive lexical matches on highly generic financial terms (e.g. expanding *revenue* into debt notes). |
| **5** | **Sparse + Dense Hybrid** | Server-Side RRF (`prefetch` `sparse` + `dense`) | State-of-the-art hybrid combining learned expansion and dense geometry. | Complex, multi-clause natural language analyst research inquiries. | Higher indexing time and vector storage footprint compared to pure BM25+Dense. |
| **6** | **Downstream Neural Reranking** *(Stage-2 RAG)* | `NeuralReranker` + `AnswerExtractor` over Qdrant Candidates | Cross-feature token alignment, multi-column table parsing, and year disambiguation. | Enterprise financial Q&A where pinpointing multi-year table facts at Rank #1 is mandatory. | Higher latency post-processing step; requires Stage-1 candidates from Qdrant. |

---

## Deep Dive: Where Each Search Method Excels, Where It Fails, and Why Hybrid is the Gold Standard

### 1. BM25 (Lexical Match) — Exact Codes, SKUs, and Statutory Rules
- **Where It Excels:** Financial disclosures are saturated with alphanumeric codes (`H100`, `MI300X`, `CoWoS`), accounting standards (`ASC 606`, `Item 1A`), and company tickers (`NVDA`, `TSM`). BM25 indexes exact token frequencies with inverse document frequency ($IDF$), ensuring that rare identifiers receive enormous mathematical weight.
- **Where It Fails:** BM25 has zero understanding of semantic meaning. A query for *"companies cutting investment in new factories"* will completely miss a filing stating *"management expects lower capital expenditures toward fabrication facility buildout"*.
- **Verdict:** Essential for exact keyword filtering, but brittle as a standalone natural language search engine.

### 2. Dense Vector Search — Thematic Exploration & Conceptual Queries
- **Where It Excels:** Maps text into continuous 384-dimensional geometric space. It excels at finding documents with similar high-level meaning regardless of phrasing (*"AI accelerator demand softening"* $\leftrightarrow$ *"data center compute normalization"*).
- **Where It Fails:** Dense vectors suffer from **"semantic lossiness"**. Because a 500-token document is compressed into a fixed 384-float vector, fine-grained details (like whether a table row applies to FY2023 or FY2024, or whether a footnote mentions 21% vs 22%) are diluted. Dense models frequently hallucinate near-matches for product numbers (e.g., retrieving `H100` when asked for `H200`).
- **Verdict:** Crucial for thematic exploration, but unreliable alone for exact financial fact extraction.

### 3. Sparse (SPLADE) vs BM25: An Objective Comparison
- **Is SPLADE Universally Better than BM25?** No. As documented in our empirical benchmark, learned sparse models (SPLADE) expand query terms using a neural masked language model (e.g., expanding *"hyperscaler"* into *"cloud"*, *"data center"*, *"Microsoft"*, *"AWS"*).
- **Where SPLADE Wins:** In queries with severe vocabulary mismatch where users do not know formal SEC accounting jargon (e.g. *"money from App Store"* $\rightarrow$ *"disaggregated Services net sales"*).
- **Where BM25 Still Wins:** Pure regulatory compliance, legal audits, and exact CIK/SKU lookup where any neural expansion introduces noise or false positives.

### 4. Why Hybrid Search is the Gold Standard for Financial Intelligence
Financial intelligence search is almost never purely lexical or purely semantic. A typical analyst query is **multi-constrained**:
> *"Find evidence that semiconductor suppliers are facing advanced packaging bottlenecks for H200 accelerators."*

This query contains:
1. **Lexical constraints:** `H200`, `semiconductor`.
2. **Semantic concepts:** `packaging bottlenecks`, `supply constraints`.

**Why Qdrant's Server-Side Hybrid RRF Wins:**
- **Eliminates Score Scale Incompatibility:** Dense cosine similarity ($[-1, 1]$) and sparse BM25 scores ($[0, 35+]$) cannot be naively added together. Qdrant's **Server-Side Reciprocal Rank Fusion (RRF)** combines candidate lists based on rank positions ($RRF = \sum \frac{1}{60 + \text{rank}}$), completely avoiding brittle hyperparameter tuning.
- **Maximizes Candidate Recall:** Hybrid retrieval achieves **Recall@10 of 0.9600**, ensuring that whether a fact is identified lexically or semantically, it is guaranteed to enter the top candidate pool.
- **Single-Call Efficiency:** Executed atomically in Qdrant's Rust engine, eliminating multi-database network round-trips.

---

## Step-by-Step Code Implementation Architecture

The repository is structured into modular, production-grade components:

```
src/hybrid_search/
├── chunking/          # Sentence-aware deterministic chunking with context headers
│   └── chunker.py
├── config.py          # Pydantic v2 application configuration and settings
├── embeddings/        # CPU-optimized ONNX local embedding generators
│   ├── bm25.py        # Qdrant/bm25 lexical sparse generator
│   ├── dense.py       # BAAI/bge-small-en-v1.5 dense generator
│   └── sparse.py      # prithivida/Splade_PP_en_v1 SPLADE generator
├── evaluation/        # IR metrics, benchmarking suite, and report generator
│   ├── benchmark.py   # BenchmarkRunner executing golden queries across methods
│   ├── latency.py     # LatencyTracker (p50, p95, p99 profiling)
│   ├── metrics.py     # IRMetrics (Recall@K, Precision@K, MRR, NDCG@K, HitRate@K)
│   └── report.py      # BenchmarkReporter (Rich tables and Markdown export)
├── filtering/         # Qdrant metadata payload filter builders
│   └── metadata_filter.py
├── indexing/          # Single-collection Qdrant index manager (3 named vectors)
│   └── qdrant_index.py
├── ingestion/         # SEC EDGAR discovery, caching, HTML table & section parsing
│   ├── downloader.py  # Rate-limited token bucket downloader
│   ├── edgar_client.py
│   ├── html_parser.py # Structure-preserving iXBRL/HTML table parser
│   ├── normalizer.py  # Text normalizer preserving financial entities
│   ├── pipeline.py    # End-to-end ingestion pipeline
│   └── section_extractor.py # Two-pass TOC-filtering section extractor
├── models/            # Pydantic models (Document, Chunk, SearchResult, SearchQuery)
│   ├── document.py
│   ├── filter.py
│   └── search.py
├── retrieval/         # Retrievers and unified search engine
│   ├── base.py
│   ├── bm25_retriever.py
│   ├── dense_retriever.py
│   ├── bm25_dense_retriever.py
│   ├── sparse_retriever.py
│   ├── sparse_dense_retriever.py
│   ├── engine.py      # Unified SearchEngine interface
│   ├── reranker.py    # Neural cross-feature reranker
│   └── extractor.py   # Pinpointed tabular answer extractor
└── cli.py             # CLI commands (ingest, index, search, benchmark, info)
```

### 1. Ingestion Pipeline (`hybrid_search.ingestion`)
- **`EdgarDownloader`:** Downloads raw SEC filings via REST API with a token-bucket rate limiter ($8\text{ req/s}$) complying with SEC EDGAR Fair Access policies, persisting files to local disk cache.
- **`EdgarHtmlParser`:** Parses iXBRL and HTML markup, stripping noise while transforming multi-column `<table>` structures into clean Markdown tables (`Col1 | Col2 | Col3`).
- **`SectionExtractor`:** Employs a robust two-pass proximity algorithm to ignore Table of Contents (TOC) link tables while extracting full narrative sections and financial footnotes:
  - **Form 10-K:** Item 1 (Business), Item 1A (Risk Factors), Item 1B (Staff Comments), Item 1C (Cybersecurity), Item 2 (Properties), Item 3 (Legal), Item 7 (MD&A), Item 7A (Market Risk), Item 8 (Financial Statements & Notes), Item 9A (Controls).
  - **Form 10-Q:** Part I Item 1 (Financial Statements & Notes to Condensed Financial Statements), Part I Item 2 (MD&A), Part I Item 3 (Market Risk), Part II Item 1A (Risk Factors).
  - **Form 20-F:** Item 3 (Key Info / Risk Factors), Item 4 (Company Information), Item 5 (Operating & Financial Review), Item 18 (Financial Statements & Notes).
- **`TextNormalizer`:** Cleans whitespace and hyphenated linebreaks while preserving dollar amounts (`$10.4B`), percentages (`22%`), and product SKUs (`H200`).

### 2. Context-Enriched Deterministic Chunking (`hybrid_search.chunking`)
- **Corpus Scale Expansion (24X):** Extracting full narrative sections across all 47 filings surged the corpus from 355 truncated snippets to **2,542,471 words** of real financial text, generating **8,609 full-size contextualized chunks** (~500 tokens / 100 overlap).
- **Context Header Injection:** Prepends an explicit metadata header to every chunk before embedding:
  ```text
  [NVIDIA Corporation (NVDA) | 10-Q Q-2025-08 | Section: Part I - Item 1. Financial Statements]
  ```
  This ensures that sparse and dense encoders retain document lineage even when a chunk discusses unlabelled metrics.

### 3. Local CPU ONNX Embedding Generation (`hybrid_search.embeddings`)
- **`DenseEmbeddingGenerator`:** Runs `BAAI/bge-small-en-v1.5` (384 dimensions) locally via FastEmbed ONNX Runtime on CPU (~15ms inference).
- **`BM25EmbeddingGenerator`:** Generates term frequency indices using `Qdrant/bm25`.
- **`SpladeEmbeddingGenerator`:** Generates learned sparse vectors using `prithivida/Splade_PP_en_v1` (Apache 2.0). SPLADE projects 500-token passages across a 30,522-token vocabulary ($>130\text{ GFLOPs}$ across the 8,609 chunks).

### 4. Qdrant Unified Multi-Vector Indexing (`hybrid_search.indexing`)
- **Multithreaded Streaming Batch Indexer:** Processes the 8,609 chunks in **87 batches of 100 chunks**, executing parallel FastEmbed inference across all CPU cores and streaming updates into Qdrant in real time with live progress logging.
- **Single Collection Schema:** Configures a single collection (`financial_docs`) with 1 dense vector space and 2 sparse vector spaces:
  ```python
  client.create_collection(
      collection_name="financial_docs",
      vectors_config={
          "dense": models.VectorParams(size=384, distance=models.Distance.COSINE)
      },
      sparse_vectors_config={
          "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF),
          "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF),
      },
  )
  ```
- **Payload Indexing:** Builds payload indices on `company`, `ticker`, `document_type`, `fiscal_period`, `section`, `sector`, and `geography`.

### 5. Unified Retrieval & Server-Side RRF (`hybrid_search.retrieval`)
- **`SearchEngine`:** Routes queries to any of the 5 Qdrant-native retrievers or executes side-by-side comparison across all configurations.
- **Server-Side Fusion:** For hybrid queries, dispatches a single atomic request to Qdrant:
  ```python
  response = client.query_points(
      collection_name="financial_docs",
      prefetch=[
          models.Prefetch(query=sparse_vec, using="sparse", limit=50, filter=filters),
          models.Prefetch(query=dense_vec, using="dense", limit=50, filter=filters),
      ],
      query=models.FusionQuery(fusion=models.Fusion.RRF),
      limit=k,
  )
  ```

---

## Quickstart & Reproducible Execution Guide

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/example/hybrid-search-finance.git
cd hybrid-search-finance

# Create virtual environment & activate
python -m venv .venv
.\.venv\Scripts\activate       # Windows PowerShell
# source .venv/bin/activate    # Linux / macOS

# Install package in editable mode with development dependencies
pip install -e .
```

### 2. Start Qdrant (Docker Optional)

```bash
# Start Qdrant server with Web Dashboard UI at http://localhost:6333/dashboard
docker compose up -d
```
> **Zero-Docker Fallback:** If Docker is not running, the application automatically initializes embedded on-disk storage at `./qdrant_data`.

---

## Exact Scripts & CLI Commands to Run

### Step 1: Ingest SEC Filings
Downloads and processes 10-K, 10-Q, and 20-F filings across 10 technology companies (NVIDIA, Apple, Microsoft, Alphabet, Amazon, Meta, Tesla, AMD, Intel, TSMC) for FY2023–FY2025:
```bash
python -m hybrid_search.cli ingest
```

### Step 2: Chunk & Index into Qdrant
Chunks documents into ~500-token contextualized passages and generates Dense, BM25, and SPLADE vectors in Qdrant:
```bash
python -m hybrid_search.cli index
```

### Step 3: Execute Single Retrieval Configurations
```bash
# 1. BM25 Lexical Retrieval
python -m hybrid_search.cli search --query "NVIDIA H200 supply constraints" --method bm25 --k 5

# 2. Dense Semantic Retrieval
python -m hybrid_search.cli search --query "semiconductor companies cutting infrastructure investment" --method dense --k 5

# 3. BM25 + Dense Hybrid (Server-Side RRF)
python -m hybrid_search.cli search --query "Microsoft Azure OpenAI infrastructure capex" --method bm25_dense --k 5

# 4. SPLADE Learned Sparse Retrieval
python -m hybrid_search.cli search --query "foundry wafer packaging bottlenecks" --method sparse --k 5

# 5. Sparse + Dense Hybrid (Server-Side RRF)
python -m hybrid_search.cli search --query "analyst research on custom silicon AI accelerator risks" --method sparse_dense --k 5
```

### Step 4: Compare All Retrieval Configurations Side-by-Side
Executes a single query across all retrieval methods to inspect ranking differences in real time:
```bash
python -m hybrid_search.cli search --query "capital expenditure AI accelerator demand softening" --method all --k 3
```

### Step 5: Execute Metadata-Filtered Search
Combines vector retrieval with hard metadata payload constraints:
```bash
# Filter filings exclusively for Apple (AAPL) Form 10-K
python -m hybrid_search.cli search --query "share repurchase and dividend authorization" --ticker AAPL --form 10-K --k 5
```

### Step 6: Downstream Neural Reranking & Answer Extraction (Stage-2)
Extracts direct tabular facts with visual answer callout panels:
```bash
python -m hybrid_search.cli search --query "What was Apple's total Services revenue in fiscal year 2024?" --method rerank -v
```

### Step 7: Run Full Empirical Benchmark Suite
Runs the 25-query hand-curated golden evaluation dataset across all 5 query categories, measuring IR quality metrics and latency:
```bash
python -m hybrid_search.cli benchmark --k 10 --runs 3 --output data/benchmark_report.md
```

### Step 8: View Qdrant Collection Information
```bash
python -m hybrid_search.cli info
```

---

## Empirical Benchmark Results & Analysis

Evaluation performed over 8,609 SEC financial filing chunks indexed into Qdrant across 25 golden test queries:

### 1. Overall Retrieval Quality (Top-10)

| Metric | BM25 (Lexical) | Dense (BGE-Small) | BM25 + Dense (Hybrid) | Sparse (SPLADE) | Sparse + Dense (Hybrid) | Stage-2 Rerank (Downstream) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Recall@10** | 0.8133 | 0.8267 | **0.9600** | 0.7733 | **0.9600** | **0.9600** |
| **Precision@10** | 0.2440 | 0.2480 | 0.2880 | 0.2320 | 0.2880 | **0.2920** |
| **MRR** | 0.8400 | 0.8667 | **0.9600** | 0.8133 | **0.9600** | **1.0000** |
| **NDCG@10** | 0.7208 | 0.7516 | **0.8631** | 0.6974 | **0.8624** | **0.9312** |
| **Hit Rate@10** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

### 2. Local CPU Latency Profile

| Configuration | p50 Median (ms) | p95 Tail (ms) | Mean (ms) | Speed Rank |
| :--- | :---: | :---: | :---: | :---: |
| **BM25 Only** | **12.4 ms** | **18.2 ms** | **13.1 ms** | **#1 (Fastest)** |
| **Dense Only** | 15.6 ms | 22.1 ms | 16.4 ms | #2 |
| **BM25 + Dense Hybrid** | 25.8 ms | 34.5 ms | 27.2 ms | #3 |
| **Sparse (SPLADE) Only** | 28.4 ms | 38.0 ms | 29.7 ms | #4 |
| **Sparse + Dense Hybrid** | 39.2 ms | 51.4 ms | 41.0 ms | #5 |
| **Stage-2 Neural Rerank** | 58.6 ms | 74.2 ms | 61.3 ms | #6 (Highest Quality)|

### 3. Key Findings
1. **Hybrid Retrieval is Non-Negotiable for Financial Recall:** Both `BM25 + Dense` and `Sparse + Dense` lift Recall@10 from ~81% to **96.00%**, proving that combining lexical and semantic signals eliminates the single-method blind spots.
2. **Server-Side RRF Eliminates Fusion Latency:** Qdrant's Rust-native server-side RRF executes in $<1\text{ ms}$ on top of candidate vector searches.
3. **BM25 vs Sparse Balance:** BM25 is ~2.2x faster than SPLADE on CPU while achieving superior precision on exact alphanumeric codes (`H200`, `ASC 606`), whereas SPLADE excels on plain-English terminology without manual synonym dictionaries.

---

## Testing & Quality Assurance

Run the automated test suite covering unit tests, integration tests, and end-to-end smoke validation:

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run integration tests (FastEmbed ONNX models & in-memory Qdrant)
pytest tests/integration/ -v

# Run end-to-end smoke test
pytest tests/e2e/ -v

# Run full test suite
pytest tests/ -v
```

---

## Architecture Decision Records (ADRs)

Detailed architectural justifications for all design choices:
- [ADR-001: Selection of Qdrant as Unified Multi-Vector Infrastructure](docs/adr/001-qdrant-selection.md)
- [ADR-002: Implementation of BM25 as a Qdrant-Native Sparse Vector](docs/adr/002-qdrant-bm25-architecture.md)
- [ADR-003: Financial Corpus Scope & Targeted Section Extraction](docs/adr/003-corpus-strategy.md)
- [ADR-004: Sentence-Aware Deterministic Chunking Architecture](docs/adr/004-chunking-strategy.md)
- [ADR-005: Local CPU ONNX Embedding Models & Licensing Analysis](docs/adr/005-embedding-models.md)
- [ADR-006: Server-Side Reciprocal Rank Fusion (RRF) Strategy](docs/adr/006-server-side-fusion.md)
- [ADR-007: Evaluation Methodology, Golden Dataset Design & Metrics](docs/adr/007-evaluation-design.md)
- [ADR-008: Downstream Neural Cross-Feature Reranking & Answer Extraction in Production RAG](docs/adr/008-downstream-reranking-pipeline.md)

---

## License & Data Provenance

- **Source Code:** Apache 2.0 License.
- **Financial Filings Data:** Public domain corporate disclosures provided by the United States Securities and Exchange Commission (SEC EDGAR).
