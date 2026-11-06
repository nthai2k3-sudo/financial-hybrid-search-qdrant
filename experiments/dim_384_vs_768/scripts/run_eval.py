from __future__ import annotations

import json
import time
from pathlib import Path

from common import (
    AUDIT,
    METHODS,
    RAW,
    SOURCE_COLLECTION,
    TARGET_COLLECTION,
    experiment_config,
    load_queries,
    metrics_for_results,
    open_local_client,
)
from hybrid_search.retrieval.engine import SearchEngine


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    client = open_local_client()
    for label, collection, model in [
        ("384", SOURCE_COLLECTION, "BAAI/bge-small-en-v1.5"),
        ("768", TARGET_COLLECTION, "BAAI/bge-base-en-v1.5"),
    ]:
        config = experiment_config(collection, model)
        engine = SearchEngine(config=config, client=client)
        output = {
            "collection": collection,
            "dense_model": model,
            "k": 10,
            "raw_top_k": 20,
            "queries": [],
        }
        for query in load_queries():
            query_output = {
                "id": query["id"],
                "category": query.get("category", "general"),
                "query": query["query"],
                "relevant_terms": query.get("relevant_terms", []),
                "methods": {},
            }
            for method in METHODS:
                started = time.perf_counter()
                results = engine.search(query["query"], method=method, k=20, prefetch_k=50)
                elapsed_ms = (time.perf_counter() - started) * 1000
                # Qdrant may return exact-score ties in storage-dependent order.
                # Make the order reproducible without changing scores or membership.
                results = sorted(results, key=lambda item: (-item.score, item.chunk_id))
                serialized = [
                    {
                        "chunk_id": result.chunk_id,
                        "score": result.score,
                        "rank": rank,
                        "text": result.text,
                        "ticker": result.ticker,
                        "document_type": result.document_type,
                        "fiscal_period": result.fiscal_period,
                        "section": result.section,
                    }
                    for rank, result in enumerate(results, start=1)
                ]
                query_output["methods"][method] = {
                    "elapsed_ms": elapsed_ms,
                    "results": serialized,
                    "metrics": metrics_for_results(
                        serialized, query.get("relevant_terms", [])
                    ),
                }
            output["queries"].append(query_output)
        path = RAW / f"{label}.json"
        path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote={path}")
    client.close()


if __name__ == "__main__":
    main()
