# System Architecture Documentation

This document provides a comprehensive technical overview of the Financial Intelligence Hybrid Search reference architecture on **Qdrant Vector Database**, including system component layouts, data flows for the 5 Qdrant-native retrieval methods and downstream Stage-2 reranking, module dependencies, Qdrant collection specifications, and evaluation methodology.

---

## 1. High-Level System Architecture

```mermaid
graph TB
    subgraph Ingestion["1. SEC EDGAR Ingestion Pipeline"]
        EDGAR["SEC EDGAR Submissions API<br/>(data.sec.gov)"] --> Downloader["EdgarDownloader<br/>(Token Bucket 8 req/s, Cache)"]
        Downloader --> HTMLParser["EdgarHtmlParser<br/>(iXBRL Stripping, Table Parsing)"]
        HTMLParser --> Extractor["SectionExtractor<br/>(Item 1 Business, Item 1A Risk, Item 7/2 MD&A, Item 8/1 Notes)"]
        Extractor --> Normalizer["TextNormalizer<br/>(Financial Entity & Symbol Preservation)"]
        Normalizer --> DocJSONL[("Processed Documents<br/>(data/processed/documents.jsonl)")]
    end

    subgraph Chunking["2. Deterministic Chunking Engine"]
        DocJSONL --> Chunker["DocumentChunker<br/>(~500 Tokens / 100 Overlap, Context-Enriched Headers)"]
        Chunker --> ChunkJSONL[("Processed Chunks<br/>(data/processed/chunks.jsonl)")]
    end

    subgraph Embedding["3. Local CPU ONNX Embedding Generation"]
        ChunkJSONL --> DenseGen["DenseEmbeddingGenerator<br/>(BAAI/bge-small-en-v1.5, 384d)"]
        ChunkJSONL --> BM25Gen["BM25EmbeddingGenerator<br/>(Qdrant/bm25 Lexical Sparse)"]
        ChunkJSONL --> SpladeGen["SpladeEmbeddingGenerator<br/>(prithivida/Splade_PP_en_v1)"]
    end

    subgraph Storage["4. Qdrant Unified Multi-Vector Storage"]
        DenseGen --> QdrantIndex["QdrantIndexManager<br/>(Collection: financial_docs)"]
        BM25Gen --> QdrantIndex
        SpladeGen --> QdrantIndex
        
        QdrantIndex --> Col[("Single Unified Qdrant Collection<br/>• Named Vector: 'dense' (384d Cosine)<br/>• Named Vector: 'bm25' (Sparse Modifier.IDF)<br/>• Named Vector: 'sparse' (SPLADE Modifier.IDF)<br/>• Indexed Payload Schema")]
    end

    subgraph Retrieval["5. Qdrant-Native Retrieval Layer (5 Core Configurations)"]
        Col --> R1["1. BM25Retriever (Lexical Sparse)"]
        Col --> R2["2. DenseRetriever (Semantic Cosine)"]
        Col --> R3["3. BM25DenseRetriever (Server-Side RRF)"]
        Col --> R4["4. SparseRetriever (SPLADE Neural Sparse)"]
        Col --> R5["5. SparseDenseRetriever (Server-Side RRF)"]
    end

    subgraph ProductionRAG["6. Downstream Production RAG Extension (Stage-2)"]
        R3 & R5 -. "High-Recall Candidates (k=20..50)" .-> Rerank["NeuralReranker<br/>(Cross-Feature Scoring)"]
        Rerank --> AnswerExt["AnswerExtractor<br/>(Table Column Alignment & Fact Pinpointing)"]
    end

    subgraph Evaluation["7. Evaluation & Benchmarking Framework"]
        R1 & R2 & R3 & R4 & R5 & Rerank --> Runner["BenchmarkRunner<br/>(golden_dataset.yaml)"]
        Runner --> Metrics["IRMetrics<br/>(Recall, Precision, MRR, NDCG, Hit Rate)"]
        Runner --> Latency["LatencyTracker<br/>(p50, p95, p99 ms)"]
        Metrics & Latency --> Reporter["BenchmarkReporter<br/>(Rich Tables & Markdown Export)"]
    end
```

---

## 2. Data Flow Diagrams for Retrieval Configurations

### 2.1 Configuration 1: BM25 (Lexical Sparse Vector)

Pure exact-match lexical retrieval using the `Qdrant/bm25` model with server-side `Modifier.IDF`.

```mermaid
sequenceDiagram
    autonumber
    actor User as Analyst / User
    participant CLI as Search Engine CLI
    participant BM25Gen as BM25EmbeddingGenerator
    participant Qdrant as Qdrant Database ('bm25' Vector Space)

    User->>CLI: search(query="NVIDIA H200 supply constraints", method="bm25", k=10)
    CLI->>BM25Gen: embed_query(query)
    Note over BM25Gen: Tokenizes query and generates term frequency indices & values
    BM25Gen-->>CLI: SparseVector(indices=[142, 8092, ...], values=[1.0, 1.0, ...])
    CLI->>Qdrant: query_points(collection="financial_docs", query=SparseVector, using="bm25", limit=10, query_filter=Filter)
    Note over Qdrant: Computes BM25 score with server-side dynamic IDF weighting
    Qdrant-->>CLI: QueryResponse(points=[ScoredPoint, ...])
    CLI-->>User: SearchResult[] (Ranked by lexical BM25 score)
```

---

### 2.2 Configuration 2: Dense (Semantic Vector)

Semantic retrieval utilizing `BAAI/bge-small-en-v1.5` 384-dimensional dense vectors.

```mermaid
sequenceDiagram
    autonumber
    actor User as Analyst / User
    participant CLI as Search Engine CLI
    participant DenseGen as DenseEmbeddingGenerator
    participant Qdrant as Qdrant Database ('dense' Vector Space)

    User->>CLI: search(query="semiconductor capex reduction", method="dense", k=10)
    CLI->>DenseGen: embed_query(query)
    Note over DenseGen: Computes 384-d dense embedding via FastEmbed ONNX
    DenseGen-->>CLI: dense_vector [0.042, -0.018, ..., 0.089]
    CLI->>Qdrant: query_points(collection="financial_docs", query=dense_vector, using="dense", limit=10, query_filter=Filter)
    Note over Qdrant: HNSW Cosine Similarity Search
    Qdrant-->>CLI: QueryResponse(points=[ScoredPoint, ...])
    CLI-->>User: SearchResult[] (Ranked by Cosine Similarity)
```

---

### 2.3 Configuration 3: BM25 + Dense Hybrid (Server-Side RRF)

Combines exact lexical sparse matching with dense semantic geometry in a single server-side call using Qdrant's `prefetch` and `FusionQuery(fusion=Fusion.RRF)`.

```mermaid
sequenceDiagram
    autonumber
    actor User as Analyst / User
    participant Engine as SearchEngine
    participant BM25Gen as BM25EmbeddingGenerator
    participant DenseGen as DenseEmbeddingGenerator
    participant Qdrant as Qdrant Server-Side Engine

    User->>Engine: search(query="Microsoft Azure OpenAI capex", method="bm25_dense", k=10)
    par Generate Sparse & Dense Vectors
        Engine->>BM25Gen: embed_query(query)
        BM25Gen-->>Engine: bm25_sparse_vector
    and
        Engine->>DenseGen: embed_query(query)
        DenseGen-->>Engine: dense_vector
    end
    Engine->>Qdrant: query_points(<br/>  prefetch=[<br/>    Prefetch(query=bm25_sparse_vector, using="bm25", limit=50),<br/>    Prefetch(query=dense_vector, using="dense", limit=50)<br/>  ],<br/>  query=FusionQuery(fusion=Fusion.RRF),<br/>  limit=10<br/>)
    Note over Qdrant: 1. Executes BM25 sparse search (limit 50)<br/>2. Executes Dense cosine search (limit 50)<br/>3. Applies Reciprocal Rank Fusion: RRF = Σ 1/(60 + rank)
    Qdrant-->>Engine: QueryResponse(points=[ScoredPoint with RRF score, ...])
    Engine-->>User: SearchResult[] (Fused Hybrid Results)
```

---

### 2.4 Configuration 4: Sparse Retrieval (SPLADE Learned Sparse)

Learned sparse neural retrieval utilizing `prithivida/Splade_PP_en_v1` with neural term expansion over a 30k BERT vocabulary.

```mermaid
sequenceDiagram
    autonumber
    actor User as Analyst / User
    participant CLI as Search Engine CLI
    participant SpladeGen as SpladeEmbeddingGenerator
    participant Qdrant as Qdrant Database ('sparse' Vector Space)

    User->>CLI: search(query="foundry wafer bottlenecks", method="sparse", k=10)
    CLI->>SpladeGen: embed_query(query)
    Note over SpladeGen: Generates learned sparse vector with neural expansion (e.g. activates 'fabrication', 'capacity', 'TSMC')
    SpladeGen-->>CLI: SparseVector(indices=[452, 1089, ...], values=[2.14, 1.87, ...])
    CLI->>Qdrant: query_points(collection="financial_docs", query=SparseVector, using="sparse", limit=10, query_filter=Filter)
    Note over Qdrant: Sparse inverted index search with Modifier.IDF
    Qdrant-->>CLI: QueryResponse(points=[ScoredPoint, ...])
    CLI-->>User: SearchResult[] (Ranked by SPLADE score)
```

---

### 2.5 Configuration 5: Sparse + Dense Hybrid (Server-Side RRF)

State-of-the-art hybrid search combining neural expansion sparse vectors with dense semantic embeddings, fused atomically inside Qdrant.

```mermaid
sequenceDiagram
    autonumber
    actor User as Analyst / User
    participant Engine as SearchEngine
    participant SpladeGen as SpladeEmbeddingGenerator
    participant DenseGen as DenseEmbeddingGenerator
    participant Qdrant as Qdrant Server-Side Engine

    User->>Engine: search(query="analyst research on AI hardware risks", method="sparse_dense", k=10)
    par Generate SPLADE & Dense Vectors
        Engine->>SpladeGen: embed_query(query)
        SpladeGen-->>Engine: splade_sparse_vector
    and
        Engine->>DenseGen: embed_query(query)
        DenseGen-->>Engine: dense_vector
    end
    Engine->>Qdrant: query_points(<br/>  prefetch=[<br/>    Prefetch(query=splade_sparse_vector, using="sparse", limit=50),<br/>    Prefetch(query=dense_vector, using="dense", limit=50)<br/>  ],<br/>  query=FusionQuery(fusion=Fusion.RRF),<br/>  limit=10<br/>)
    Note over Qdrant: Server-side RRF merges neural sparse and dense candidate sets
    Qdrant-->>Engine: QueryResponse(points=[ScoredPoint with RRF score, ...])
    Engine-->>User: SearchResult[] (Ranked by Combined RRF Score)
```

---

### 2.6 Downstream Stage-2 Extension: Neural Cross-Feature Reranking & Answer Extraction

Demonstrates how downstream RAG applications consume Qdrant's high-recall candidate pool for exact tabular and temporal precision.

```mermaid
sequenceDiagram
    autonumber
    actor User as Analyst / LLM Prompt
    participant Engine as SearchEngine
    participant Qdrant as Qdrant Server-Side Engine
    participant Reranker as NeuralReranker
    participant Extractor as AnswerExtractor

    User->>Engine: search(query="What was Apple's total Services revenue in FY2024?", method="rerank", k=5)
    Engine->>Qdrant: query_points(prefetch=[sparse, dense], fusion=RRF, limit=20)
    Qdrant-->>Engine: Top 20 Candidates (High Recall Pool)
    Engine->>Reranker: rerank(query, candidates, top_k=5)
    loop For Each Candidate Chunk
        Reranker->>Extractor: extract(query, chunk_text)
        Note over Extractor: 1. Aligns multi-column table cells to FY2024<br/>2. Resolves footnote numbers<br/>3. Highlights exact clause
        Extractor-->>Reranker: ExtractedAnswer(exact_answer, confidence)
        Note over Reranker: Calculates cross-feature score (metric bonus, filing type score, coverage)
    end
    Reranker-->>Engine: SearchResult[] (Rank 1: Precision Fact)
    Engine-->>User: Top-ranked SearchResult with highlighted answer
```

---

## 3. Module Dependency Graph

```mermaid
graph TD
    subgraph Models["hybrid_search.models"]
        DocMod["document.py<br/>(Document, DocumentSection, Chunk)"]
        FiltMod["filter.py<br/>(MetadataFilter)"]
        SearchMod["search.py<br/>(SearchResult, SearchQuery)"]
    end

    subgraph Config["hybrid_search.config"]
        CfgMod["config.py<br/>(AppConfig, load_config)"]
    end

    subgraph Ingestion["hybrid_search.ingestion"]
        EdgarCl["edgar_client.py"]
        Downloader["downloader.py"]
        HtmlPars["html_parser.py"]
        SecExt["section_extractor.py"]
        Norm["normalizer.py"]
        Pipe["pipeline.py"]
    end

    subgraph Chunking["hybrid_search.chunking"]
        ChunkerMod["chunker.py"]
    end

    subgraph Embeddings["hybrid_search.embeddings"]
        DenseEmb["dense.py"]
        BM25Emb["bm25.py"]
        SpladeEmb["sparse.py"]
    end

    subgraph Indexing["hybrid_search.indexing"]
        QIndex["qdrant_index.py"]
    end

    subgraph Filtering["hybrid_search.filtering"]
        MetaFilt["metadata_filter.py"]
    end

    subgraph Retrieval["hybrid_search.retrieval"]
        BaseRet["base.py"]
        BM25Ret["bm25_retriever.py"]
        DenseRet["dense_retriever.py"]
        BM25DenseRet["bm25_dense_retriever.py"]
        SparseRet["sparse_retriever.py"]
        SparseDenseRet["sparse_dense_retriever.py"]
        RerankMod["reranker.py"]
        ExtractMod["extractor.py"]
        EngMod["engine.py"]
    end

    subgraph Evaluation["hybrid_search.evaluation"]
        MetricsMod["metrics.py"]
        LatMod["latency.py"]
        BenchMod["benchmark.py"]
        RepMod["report.py"]
    end

    subgraph CLI["hybrid_search.cli"]
        CLIMod["cli.py"]
    end

    %% Dependencies
    CfgMod --> Pipe & QIndex & EngMod & BenchMod & CLIMod
    DocMod --> Pipe & ChunkerMod & QIndex
    FiltMod --> MetaFilt & BaseRet & EngMod
    SearchMod --> BaseRet & EngMod

    EdgarCl & Downloader & HtmlPars & SecExt & Norm --> Pipe
    Pipe --> ChunkerMod
    ChunkerMod --> QIndex

    DenseEmb & BM25Emb & SpladeEmb --> QIndex & EngMod
    MetaFilt --> BM25Ret & DenseRet & BM25DenseRet & SparseRet & SparseDenseRet

    BaseRet --> BM25Ret & DenseRet & BM25DenseRet & SparseRet & SparseDenseRet
    ExtractMod --> RerankMod
    RerankMod --> EngMod
    BM25Ret & DenseRet & BM25DenseRet & SparseRet & SparseDenseRet --> EngMod

    EngMod --> BenchMod & CLIMod
    MetricsMod & LatMod & RepMod --> BenchMod
```

---

## 4. Qdrant Collection & Payload Design

### 4.1 Named Vectors Configuration
All embeddings for each chunk live within one single point in collection `financial_docs`:

```python
vectors_config = {
    "dense": models.VectorParams(
        size=384,
        distance=models.Distance.COSINE,
    )
}
sparse_vectors_config = {
    "bm25": models.SparseVectorParams(
        modifier=models.Modifier.IDF,
        index=models.SparseIndexParams(on_disk=False),
    ),
    "sparse": models.SparseVectorParams(
        modifier=models.Modifier.IDF,
        index=models.SparseIndexParams(on_disk=False),
    ),
}
```

### 4.2 Payload Indexing
Payload schema fields indexed for filtered candidate selection:
- `company` (`KEYWORD`): Full legal name of filer.
- `ticker` (`KEYWORD`): Ticker symbol (`NVDA`, `AAPL`, `MSFT`, etc.).
- `document_type` (`KEYWORD`): Form type (`10-K`, `10-Q`, `20-F`).
- `filing_date` (`DATETIME`): ISO 8601 filing date for range queries.
- `fiscal_period` (`KEYWORD`): Fiscal period label (`FY2024`, `Q3 FY2024`).
- `section` (`KEYWORD`): Item header code.
- `sector` (`KEYWORD`): Industry categorization.
- `geography` (`KEYWORD`): Geographic jurisdiction (`US`, `Taiwan`).

---

## 5. Evaluation Methodology & Metrics

### 5.1 Information Retrieval Metrics

1. **Recall@K:** Fraction of all relevant items retrieved in top-K:
   $$\text{Recall@K} = \frac{|\text{Relevant} \cap \text{Retrieved@K}|}{|\text{Relevant}|}$$

2. **Precision@K:** Fraction of retrieved items in top-K that are relevant:
   $$\text{Precision@K} = \frac{|\text{Relevant} \cap \text{Retrieved@K}|}{K}$$

3. **Mean Reciprocal Rank (MRR):** Quality of ranking the first relevant document:
   $$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_{\text{first}}(q)}$$

4. **Normalized Discounted Cumulative Gain (NDCG@K):** Graded relevance ranking metric:
   $$\text{DCG@K} = \sum_{i=1}^K \frac{rel_i}{\log_2(i+1)}, \quad \text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

5. **Hit Rate@K:** Proportion of queries where at least one relevant document appears in top-K:
   $$\text{HitRate@K} = \mathbb{I}(|\text{Relevant} \cap \text{Retrieved@K}| > 0)$$

### 5.2 Latency Profiling
Recorded in microseconds with millisecond reporting across repeated executions:
- $p_{50}$ (Median execution time)
- $p_{95}$ (95th percentile tail latency)
- $p_{99}$ (99th percentile peak latency)
- Mean and standard deviation
