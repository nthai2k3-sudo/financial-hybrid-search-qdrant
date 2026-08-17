# ADR-005: Local CPU-Optimized ONNX Embedding Models & Licensing Analysis

## Status
Accepted

## Context & Problem Statement
A hard constraint for this reference implementation is **zero paid API dependencies** (no OpenAI `text-embedding-3`, no Cohere `embed-v3`, no hosted endpoints). The entire system must run locally on standard CPU hardware with fast inference (~15–30ms per query) while maintaining state-of-the-art retrieval accuracy on technical/financial text.

Furthermore, model licensing must be strictly permissive (MIT or Apache 2.0) to permit open commercial publication and public GitHub distribution.

## Considered Models & License Analysis

### Dense Embedding Models

| Model | Dims | License | MTEB Retrieval Avg | Disk Size | CPU Latency (p50) | Assessment |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **`BAAI/bge-small-en-v1.5`** | **384** | **MIT** | **51.68** | **~67 MB** | **~12 ms** | **Selected.** Optimal quality-to-speed ratio for CPU execution; compact 384d vectors minimize Qdrant RAM/disk footprint. |
| `BAAI/bge-base-en-v1.5` | 768 | MIT | 53.25 | ~220 MB | ~35 ms | Good quality, but 2x latency and 2x vector memory over bge-small. |
| `all-MiniLM-L6-v2` | 384 | Apache 2.0 | 41.95 | ~80 MB | ~10 ms | Fast, but significantly lower retrieval accuracy on technical domain text. |
| `text-embedding-3-small` | 1536 | Proprietary | 55.40 | API-only | ~150 ms (Network) | Violates zero paid API constraint. |

### Sparse Neural Embedding Models (SPLADE)

| Model | Vocabulary | License | Commercial Permissibility | Notes |
|---|:---:|:---:|:---:|---|
| **`prithivida/Splade_PP_en_v1`** | **30,522 (BERT)** | **Apache 2.0** | **Permitted** | **Selected.** State-of-the-art SPLADE++ architecture trained with neural expansion under a permissive open-source license. |
| `naver/splade-v3` | 30,522 | CC-BY-NC-SA 4.0 | **Non-Commercial Only** | Strict non-commercial restriction makes it unsuitable for open production collaboration. |
| `naver/splade_v2_max` | 30,522 | CC-BY-NC 4.0 | **Non-Commercial Only** | Strict non-commercial restriction. |
| `Qdrant/bm42` | 30,522 | Apache 2.0 | Permitted | Heuristic attention-weight sparse model; good alternative but SPLADE is standard for learned neural expansion comparison. |

## Decision Outcome
Adopt **`BAAI/bge-small-en-v1.5` (Dense)** and **`prithivida/Splade_PP_en_v1` (Learned Sparse)** executed via **`fastembed`** (Qdrant's lightweight ONNX runtime).

### Key Architectural Benefits:
1. **Elimination of PyTorch / CUDA Overhead:** `fastembed` uses ONNX Runtime directly in C++, reducing environment footprint by ~3 GB compared to standard PyTorch+Transformers wheels.
2. **Native Output Alignment:** FastEmbed generates sparse index/value pairs that map directly to Qdrant's `models.SparseVector(indices=..., values=...)` format without custom translation layers.
3. **100% Permissive Open Source:** All models and libraries carry MIT or Apache 2.0 licenses.
