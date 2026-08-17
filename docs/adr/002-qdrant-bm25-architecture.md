# ADR-002: Implementation of BM25 as a Qdrant-Native Sparse Vector

## Status
Accepted

## Context & Problem Statement
The benchmark requires evaluating a pure lexical retrieval baseline (BM25) alongside semantic dense retrieval and learned sparse neural retrieval (SPLADE). 

Traditionally, BM25 is executed via standalone Python libraries (e.g. `rank_bm25`, `bm25s`) or Lucene-based engines (Elasticsearch). However, operating BM25 as an external system introduces major architectural hurdles:
1. Two disparate indexes on disk with distinct serialization formats.
2. Inability to execute atomic server-side Reciprocal Rank Fusion (RRF) for the `BM25 + Dense` configuration without fetching large intermediate result sets back into Python memory.
3. Inconsistent payload filtering mechanisms between Qdrant payload filters and Python in-memory post-filtering.

## Considered Alternatives

1. **Standalone `bm25s` / `rank_bm25` in Python:**
   - *Pros:* Classical implementation, transparent score inspection.
   - *Cons:* Requires client-side fusion code for `BM25 + Dense`; separate index serialization files; metadata filtering must be implemented twice; adds `scipy` dependency.
2. **Dedicated Elasticsearch / Lucene Instance:**
   - *Pros:* Industry-standard BM25 implementation.
   - *Cons:* Requires managing a heavy Java/JVM server process alongside Qdrant; complicates local setup and Docker requirements; breaks zero-overhead developer quickstart.
3. **Qdrant-Native Sparse BM25 (`Qdrant/bm25` + `Modifier.IDF`):**
   - *Pros:* Fully native to Qdrant; stores lexical term frequencies directly in Qdrant's sparse vector index; applies standard Okapi BM25 inverse document frequency dynamically server-side via `Modifier.IDF`; enables single-call server-side RRF for both hybrid configs.
   - *Cons:* IDF weights are computed within Qdrant's collection shard rather than a static global dictionary.

## Decision Outcome
Adopt **`Qdrant/bm25` sparse vectors** via FastEmbed indexed directly into Qdrant's `bm25` sparse vector space.

### Technical Mechanics:
- **Tokenization & Term Frequency (Index/Query Time):** `BM25EmbeddingGenerator` tokenizes the input text into sparse indices and token frequencies.
- **IDF Weighting (Query Time):** Qdrant's collection is configured with `models.SparseVectorParams(modifier=models.Modifier.IDF)`. During query scoring, Qdrant applies standard BM25 inverse document frequency calculations against the sparse inverted index.
- **Pure Lexical Behavior:** Unlike SPLADE, `Qdrant/bm25` does **not** perform neural term expansion. A document containing *"capital expenditure"* will **never** match a query for *"capex"* unless the exact token appears. This guarantees that the lexical vs. learned sparse distinction remains mathematically pure for benchmark validity.

## Consequences & Impact
- Single Qdrant collection holds all 3 vector representations (`dense`, `bm25`, `sparse`).
- Eliminates external dependencies (`bm25s`, `scipy`).
- Both `BM25 + Dense` and `Sparse + Dense` execute identically via Qdrant's native server-side `FusionQuery(fusion=Fusion.RRF)`.
