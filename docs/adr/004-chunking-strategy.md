# ADR-004: Sentence-Aware Deterministic Chunking Architecture

## Status
Accepted

## Context & Problem Statement
In financial intelligence search, chunking is a load-bearing architectural decision:
- Chunks that are too small (<100 tokens) lose critical context (e.g. cutting off a numerical figure from its qualifying accounting note).
- Chunks that are too large (>1000 tokens) dilute semantic embedding density, causing dense vector cosine scores to drop significantly.
- Arbitrary character-based splits cut sentences, numbers, product names (e.g. `H200` into `H2` and `00`), or dates in half.
- Non-deterministic chunking creates irreproducible vector point IDs and prevents cross-run result verification.

## Considered Strategies

1. **Fixed Character Chunking (e.g. 2000 chars, 200 overlap):**
   - *Pros:* Simple.
   - *Cons:* Splits words and sentences mid-token; destroys semantic cohesion.
2. **Recursive Character Chunking (LangChain style):**
   - *Pros:* Popular in demos.
   - *Cons:* Variable chunk sizes; non-standard token accounting; introduces unnecessary framework dependencies.
3. **Semantic Chunking (Embedding similarity split):**
   - *Pros:* Dynamically splits at semantic topic boundaries.
   - *Cons:* Requires a secondary embedding model pass per paragraph; extremely slow on CPU; non-deterministic across hardware architectures.
4. **Sentence-Aware Token Window Chunking (~500 tokens, 100 token overlap):**
   - *Pros:* Respects sentence boundaries; keeps numerical and accounting qualifiers intact; matches the optimal 512-token context window of `bge-small` and `SPLADE`; deterministic.
   - *Cons:* Chunks vary slightly in exact token counts depending on sentence lengths.

## Decision Outcome
Implement **Sentence-Aware Token Window Chunking** in [`DocumentChunker`](../../src/hybrid_search/chunking/chunker.py):

### Parameters:
- **Target Chunk Size:** 500 tokens (~375 words).
- **Chunk Overlap:** 100 tokens (~75 words).
- **Sentence Boundary Rule:** Accumulate complete sentences up to the target token threshold. When the limit is reached, emit the chunk and carry over the trailing sentences that fit within the 100-token overlap budget into the next chunk.
- **Oversized Sentence Fallback:** Single sentences longer than 1.5x target word size are split along sliding word windows to prevent memory blowup.

### Deterministic Integrity & Lineage:
Each chunk is assigned:
- **Deterministic ID:** `f"{document_id}_chunk_{chunk_index:04d}"` (e.g. `0001045810_0001045810-24-000029_chunk_0012`).
- **Integrity Hash:** SHA-256 cryptographic hash of the exact UTF-8 text content.
- **Payload Inheritance:** Every chunk inherits the parent document's `company`, `ticker`, `cik`, `document_type`, `filing_date`, `fiscal_period`, `section`, `sector`, `geography`, and `source_url`.
