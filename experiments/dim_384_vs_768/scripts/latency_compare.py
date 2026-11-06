from __future__ import annotations

import json
import platform
import statistics
import time

from common import AUDIT, METHODS, SOURCE_COLLECTION, TARGET_COLLECTION, experiment_config, load_queries, open_local_client
from hybrid_search.retrieval.engine import SearchEngine


def main() -> None:
    observations = []
    client = open_local_client()
    engines = {
        "384": SearchEngine(experiment_config(SOURCE_COLLECTION, "BAAI/bge-small-en-v1.5"), client=client),
        "768": SearchEngine(experiment_config(TARGET_COLLECTION, "BAAI/bge-base-en-v1.5"), client=client),
    }
    # BM25 and SPLADE are identical controls; share their loaded ONNX sessions
    # to keep the two-dimensional comparison within the available RAM.
    engines["384"].bm25_gen.model
    engines["384"].sparse_gen.model
    engines["768"].bm25_gen._model = engines["384"].bm25_gen._model
    engines["768"].sparse_gen._model = engines["384"].sparse_gen._model
    timers = {label: instrument(engines[label]) for label in engines}
    for query_index, query in enumerate(load_queries()):
        order = ["384", "768"] if query_index % 2 == 0 else ["768", "384"]
        for label in order:
            for method in METHODS:
                engines[label].search(query["query"], method=method, k=10, prefetch_k=50)
        for repetition in range(20):
            for label in order:
                for method in METHODS:
                    engine = engines[label]
                    timer = timers[label]
                    timer["encoding_ms"] = 0.0
                    timer["qdrant_ms"] = 0.0
                    started = time.perf_counter()
                    result = engine.search(query["query"], method=method, k=10, prefetch_k=50)
                    total_ms = (time.perf_counter() - started) * 1000
                    observations.append(
                        {
                            "collection": label,
                            "method": method,
                            "query_id": query["id"],
                            "repetition": repetition,
                            "total_ms": total_ms,
                            "encoding_ms": timer["encoding_ms"],
                            "qdrant_ms": timer["qdrant_ms"],
                            "result_count": len(result),
                        }
                    )
        print(f"latency_queries={query_index + 1}/25", flush=True)
    summary = {}
    for key in sorted({(x["collection"], x["method"]) for x in observations}):
        values = [x["total_ms"] for x in observations if (x["collection"], x["method"]) == key]
        summary[f"{key[0]}_{key[1]}"] = {
            "total_p50_ms": percentile(values, 50),
            "total_p95_ms": percentile(values, 95),
            "total_p99_ms": percentile(values, 99),
            "encoding_p50_ms": percentile([x["encoding_ms"] for x in observations if (x["collection"], x["method"]) == key], 50),
            "encoding_p95_ms": percentile([x["encoding_ms"] for x in observations if (x["collection"], x["method"]) == key], 95),
            "encoding_p99_ms": percentile([x["encoding_ms"] for x in observations if (x["collection"], x["method"]) == key], 99),
            "qdrant_p50_ms": percentile([x["qdrant_ms"] for x in observations if (x["collection"], x["method"]) == key], 50),
            "qdrant_p95_ms": percentile([x["qdrant_ms"] for x in observations if (x["collection"], x["method"]) == key], 95),
            "qdrant_p99_ms": percentile([x["qdrant_ms"] for x in observations if (x["collection"], x["method"]) == key], 99),
            "n": len(values),
        }
    payload = {
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": __import__("os").cpu_count(),
        },
        "repetitions_per_query": 20,
        "observations": observations,
        "summary": summary,
        "note": "Timers are instrumented around existing embed_query and query_points calls; total also includes result mapping and reranking.",
    }
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "latency.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    client.close()


def instrument(engine: SearchEngine) -> dict[str, float]:
    timers = {"encoding_ms": 0.0, "qdrant_ms": 0.0}
    for generator in [engine.dense_gen, engine.bm25_gen, engine.sparse_gen]:
        original = generator.embed_query

        def timed_embed(query, _original=original):
            started = time.perf_counter()
            result = _original(query)
            timers["encoding_ms"] += (time.perf_counter() - started) * 1000
            return result

        generator.embed_query = timed_embed
    original_query_points = engine.client.query_points

    def timed_query_points(*args, **kwargs):
        started = time.perf_counter()
        result = original_query_points(*args, **kwargs)
        timers["qdrant_ms"] += (time.perf_counter() - started) * 1000
        return result

    engine.client.query_points = timed_query_points
    return timers


def percentile(values: list[float], percentile_value: int) -> float:
    if not values:
        return 0.0
    return float(statistics.quantiles(values, n=100, method="inclusive")[percentile_value - 1])


if __name__ == "__main__":
    main()
