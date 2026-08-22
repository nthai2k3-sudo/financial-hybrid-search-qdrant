# Project: Financial Intelligence Hybrid Search (Qdrant)

A production-grade reference implementation demonstrating financial document retrieval over SEC EDGAR filings (Forms 10-K, 10-Q, 20-F) using Qdrant Vector Database with zero paid API dependencies.

## Tech Stack
- **Language**: Python >= 3.10
- **Vector Database**: Qdrant (`qdrant-client >= 1.14.0`) via Docker / Cloud / Embedded local storage (`qdrant_data`)
- **Embeddings / Sparse Encoders** (Local ONNX via `fastembed >= 0.5.0`):
  - Dense: `BAAI/bge-small-en-v1.5` (384 dimensions, Cosine metric)
  - Lexical BM25: `Qdrant/bm25` (Sparse with server-side `Modifier.IDF`)
  - Neural Sparse: `prithivida/Splade_PP_en_v1` (SPLADE learned sparse with `Modifier.IDF`)
- **Data Ingestion & Parsing**: `httpx`, `beautifulsoup4`, `lxml`, `pydantic >= 2.7.0`, `pyyaml >= 6.0`
- **CLI & Output Formatting**: `rich`, `argparse`
- **Testing & Benchmarking**: `pytest`, `pytest-asyncio`, `numpy`

## Architecture & 5 Core Retrieval Configurations (+ Stage 2)
Single unified Qdrant collection (`financial_docs`) with 3 named vector spaces:
1. `bm25` - Lexical matching (`Qdrant/bm25` sparse vector with `Modifier.IDF`)
2. `dense` - Semantic cosine search (`bge-small-en-v1.5`, 384d)
3. `bm25_dense` - Server-side Reciprocal Rank Fusion (RRF, $k=60$) fusing BM25 + Dense
4. `sparse` - SPLADE learned neural sparse vector search
5. `sparse_dense` - Server-side RRF fusing SPLADE Sparse + Dense
6. `rerank` - Downstream Stage-2 Cross-Feature Neural Reranker + Pinpointed Table Answer Extractor

## Key Commands
- **CLI Entrypoint**: `python -m hybrid_search.cli` or `hybrid-search`
- **Ingest SEC Filings**: `python -m hybrid_search.cli ingest` (or `--force`)
- **Index Chunks into Qdrant**: `python -m hybrid_search.cli index` (or `--recreate`)
- **Search**: `python -m hybrid_search.cli search "<query>" --method sparse_dense -k 10`
  - Methods: `bm25`, `dense`, `bm25_dense`, `sparse`, `sparse_dense`, `rerank`, `all`
  - Filters: `--ticker NVDA`, `--form 10-K`, `--section "Item 1A"`
- **Run Benchmark**: `python -m hybrid_search.cli benchmark --output data/benchmark_report.md`
- **Run Tests**: `pytest`
  - Unit tests only: `pytest tests/unit`
  - Integration tests: `pytest tests/integration`
  - E2E tests: `pytest tests/e2e`

## Project Structure
- `src/hybrid_search/`
  - `chunking/`: Context-enriched chunking (`chunker.py`) with metadata headers
  - `config.py`: Pydantic v2 application configuration
  - `embeddings/`: Local CPU ONNX encoders (`dense.py`, `bm25.py`, `sparse.py`)
  - `evaluation/`: IR metrics (`metrics.py`), latency tracker (`latency.py`), benchmarking (`benchmark.py`), reporter (`report.py`)
  - `filtering/`: Qdrant payload metadata filter generator (`metadata_filter.py`)
  - `indexing/`: Single-collection manager (`qdrant_index.py`)
  - `ingestion/`: SEC EDGAR downloader, iXBRL table parser, section extractor, normalizer, pipeline
  - `models/`: Pydantic schemas (`document.py`, `filter.py`, `search.py`)
  - `retrieval/`: 5 retrievers, `SearchEngine`, `NeuralReranker`, `AnswerExtractor`
  - `cli.py`: Rich CLI interface
- `config/`: `default.yaml`, `companies.yaml`
- `data/`: `raw/edgar/` (cached filings), `processed/` (`documents.jsonl`, `chunks.jsonl`, `manifest.json`)
- `evaluation/`: `golden_dataset.yaml`
- `docs/`: Architecture docs and ADRs (001 to 008)
- `tests/`: Unit, integration, and e2e smoke tests

## Code & Quality Conventions
- Type annotations required on all function signatures (`from __future__ import annotations`).
- Pydantic v2 for data validation, serialization, and settings models.
- Graceful fallbacks: If Qdrant server is unreachable, falls back to embedded local storage (`qdrant_data`).
- Maintain zero external paid API dependencies: All embeddings, sparse vectors, and rerankers must run locally on CPU/ONNX.
- SEC Fair Access Compliance: Edgar downloader token bucket rate limit $\le 8\text{ req/s}$ with compliant User-Agent headers.
