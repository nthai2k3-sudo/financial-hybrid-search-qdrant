# ADR-006: Server-Side Reciprocal Rank Fusion (RRF) Strategy

## Status
Accepted

## Context & Problem Statement
Hybrid retrieval combines candidate rankings generated from heterogeneous scoring functions:
- Dense vector search scores: Cosine similarity in range $[-1.0, 1.0]$ (typically $[0.4, 0.9]$ for relevant text).
- BM25 scores: Unbounded positive floating-point numbers depending on query length and collection term frequencies ($[0.0, 35.0+]$).
- SPLADE sparse scores: Dot products of sparse expanded activations ($[0.0, 50.0+]$).

Because score scales and distributions differ drastically between dense and sparse spaces, naive linear weighted score combinations ($\alpha \cdot S_{\text{dense}} + (1-\alpha) \cdot S_{\text{sparse}}$) suffer from severe calibration drift and require fragile, query-specific hyperparameter tuning.

## Considered Fusion Algorithms

1. **Linear Weighted Score Combination:**
   - *Pros:* Simple.
   - *Cons:* Requires score normalization (e.g. Min-Max or Z-Score), which is vulnerable to extreme outlier scores; requires tuning weight $\alpha$; unstable across diverse query categories.
2. **Distribution-Based Score Fusion (DBSF):**
   - *Pros:* Normalizes scores based on statistical distribution per query.
   - *Cons:* Requires a large candidate pool ($k > 100$) to compute meaningful distribution parameters; sensitive to small candidate sets.
3. **Reciprocal Rank Fusion (RRF):**
   - *Pros:* Rank-based and score-invariant; robust across vastly different score distributions; does not require per-query score normalization; proven standard in IR benchmarks (Cormack et al.).
   - *Cons:* Ignores relative margin of victory between ranks (rank 1 vs rank 2 is treated identically regardless of raw score gap).

## Decision Outcome
Adopt **Reciprocal Rank Fusion (RRF)** executed **server-side within Qdrant**:

### Mathematical Formulation:
$$RRF\_Score(d) = \sum_{m \in M} \frac{1}{k + \text{rank}_m(d)}$$

Where:
- $M$ is the set of retrieval systems (e.g. $M = \{\text{dense}, \text{bm25}\}$ or $M = \{\text{dense}, \text{sparse}\}$).
- $\text{rank}_m(d)$ is document $d$'s 1-based rank in retrieval system $m$'s candidate list.
- $k = 60$ is the standard smoothing constant (mitigates high sensitivity to the top few ranks).

### Server-Side Implementation in Qdrant:
```python
response = client.query_points(
    collection_name="financial_docs",
    prefetch=[
        models.Prefetch(query=sparse_vector, using="sparse", limit=50, filter=filters),
        models.Prefetch(query=dense_vector, using="dense", limit=50, filter=filters),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=10,
)
```

## Architectural Consequences
- **Zero Client Network Overhead:** The candidate list retrieval, ranking, and reciprocal score accumulation execute in-memory inside Qdrant's Rust engine. Only the final top-$K$ scored points are serialized and transmitted over the network.
- **Unified for Both Hybrids:** Both `BM25 + Dense` and `Sparse + Dense` utilize the identical server-side fusion pathway.
