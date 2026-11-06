# Changes

- Added only experiment artifacts under `experiments/dim_384_vs_768/`.
- No source retrieval/evaluation behavior was changed.
- The existing default configuration and dependency changes predate this
  experiment; this experiment does not modify them.
- Raw experiment results apply a deterministic `(-score, chunk_id)` tie-break
  because the two local Qdrant collections returned exact-score BM25 ties in
  different storage order; scores and result membership were unchanged.
