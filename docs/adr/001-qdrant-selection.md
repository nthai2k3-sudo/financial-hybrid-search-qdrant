# ADR-001: Selection of Qdrant as the Unified Multi-Vector Retrieval Infrastructure

## Status
Accepted

## Context & Problem Statement
Building a financial intelligence hybrid search reference implementation requires supporting multiple vector and lexical retrieval paradigms:
1. Dense vector retrieval (Cosine similarity on high-dimensional semantic spaces).
2. Lexical keyword retrieval (Term frequency / BM25).
3. Learned sparse neural retrieval (SPLADE neural term expansion).
4. Complex payload metadata filtering (Ticker, filing type, datetime range, sector).
5. Score fusion (Reciprocal Rank Fusion) across candidate lists.

Historically, production architectures combine an inverted-index search engine (e.g. Elasticsearch or Apache Solr/Lucene) alongside a vector database (e.g. Milvus or FAISS). This dual-engine pattern introduces substantial operational friction: dual indexing pipelines, cross-system document ID synchronization, distributed transaction overhead, and client-side candidate merging penalties over high-latency network boundaries.

We required a single, unified database infrastructure that natively supports dense vectors, sparse vectors, payload filtering, and server-side fusion without requiring external services.

## Considered Options

| Engine | Named Multi-Vector Support | Native Sparse Vector Support (with IDF) | Server-Side RRF Fusion | Embedded / Local Disk Mode | Commercially Permissive License |
|---|:---:|:---:|:---:|:---:|:---:|
| **Qdrant** | **Yes** (Multiple named vectors per point) | **Yes** (Native sparse index + `Modifier.IDF`) | **Yes** (`prefetch` + `FusionQuery`) | **Yes** (Rust-bundled in Python client) | **Apache 2.0** |
| **Weaviate** | Yes (Multi-vector) | Limited (BM25 separate from vector index) | Yes (Hybrid search endpoint) | No (Requires Docker / Cloud) | BSD-3-Clause |
| **Milvus** | Yes (Multiple vectors) | Yes (Sparse vectors supported) | Yes (Hybrid search API) | Partial (Milvus Lite) | Apache 2.0 |
| **ChromaDB** | No (Single vector per collection) | No | No | Yes (DuckDB/SQLite) | Apache 2.0 |
| **FAISS** | No (Raw vector index only) | No (Dense only) | No (Client-side) | Yes (C++ in-memory) | MIT |
| **OpenSearch** | Limited (Dense k-NN plugin) | Yes (Native BM25) | Yes (Search pipelines) | No (Heavy JVM service) | Apache 2.0 |

## Decision Outcome
Adopt **Qdrant** (`qdrant-client` 1.19+ and Qdrant Server Docker `v1.14.0`) as the unified search and vector storage engine.

### Key Capabilities Utilized:
1. **Named Multi-Vector Schema:** A single point contains `dense` (384d Cosine), `bm25` (Sparse with `Modifier.IDF`), and `sparse` (SPLADE Sparse with `Modifier.IDF`) vectors, sharing identical payload metadata and point IDs.
2. **Server-Side Reciprocal Rank Fusion:** Hybrid queries use Qdrant's universal `query_points` API with multiple `prefetch` branches and `FusionQuery(fusion=Fusion.RRF)`, merging top candidates inside the Rust engine before transmitting results across the wire.
3. **Dual Deployment Modes:** Works against a full Docker container with web dashboard (`http://localhost:6333/dashboard`) and seamlessly falls back to embedded on-disk storage (`QdrantClient(path="./qdrant_data")`) for containerless local developer environments.

## Trade-offs & Mitigations
- **Trade-off:** Deep reliance on Qdrant-specific API constructs (`query_points`, `Prefetch`, `models.SparseVector`).
- **Mitigation:** Abstract all Qdrant calls behind standard `Retriever` and `SearchEngine` interfaces returning canonical `SearchResult` models, ensuring client code remains decoupled.
