from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np

from common import AUDIT, METHODS, RAW

METRICS = ["ndcg@10", "recall@10", "mrr", "precision@10", "hit_rate@10", "recall@1", "recall@3", "recall@5", "recall@20"]


def main() -> None:
    left = json.loads((RAW / "384.json").read_text(encoding="utf-8"))
    right = json.loads((RAW / "768.json").read_text(encoding="utf-8"))
    left_queries = {q["id"]: q for q in left["queries"]}
    right_queries = {q["id"]: q for q in right["queries"]}
    rows = []
    comparison = {"bm25_and_sparse_identical": True, "checks": []}
    for method in METHODS:
        for metric in METRICS:
            a = np.array([q["methods"][method]["metrics"][metric] for q in left["queries"]], dtype=float)
            b = np.array([right_queries[q["id"]]["methods"][method]["metrics"][metric] for q in left["queries"]], dtype=float)
            delta = b - a
            ci_low, ci_high = bootstrap_ci(delta, seed=20261104)
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "mean_384": float(np.mean(a)),
                    "mean_768": float(np.mean(b)),
                    "delta_768_minus_384": float(np.mean(delta)),
                    "delta_percent": _percent_change(float(np.mean(a)), float(np.mean(delta))),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "improved_queries": int(np.sum(delta > 0)),
                    "worsened_queries": int(np.sum(delta < 0)),
                    "unchanged_queries": int(np.sum(delta == 0)),
                    "sum_delta": float(np.sum(delta)),
                    "median_delta": float(np.median(delta)),
                    "sign_test_p": sign_test_p(delta),
                }
            )
        for query_id in left_queries:
            a_ids = [r["chunk_id"] for r in left_queries[query_id]["methods"][method]["results"]]
            b_ids = [r["chunk_id"] for r in right_queries[query_id]["methods"][method]["results"]]
            same = a_ids == b_ids if method in {"bm25", "sparse"} else None
            check = {"method": method, "query_id": query_id, "same_ids": same}
            comparison["checks"].append(check)
            if method in {"bm25", "sparse"} and not same:
                comparison["bm25_and_sparse_identical"] = False

    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "comparison_checks.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (AUDIT / "baseline_reproduction.json").write_text(
        json.dumps(compare_readme(left), indent=2), encoding="utf-8"
    )
    with (RAW.parent / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (AUDIT / "cross_check.json").write_text(json.dumps(cross_check(left, right), indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), **comparison}, indent=2))
    if not comparison["bm25_and_sparse_identical"]:
        raise RuntimeError("BM25 or sparse results changed between collections")


def bootstrap_ci(delta: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def sign_test_p(delta: np.ndarray) -> float:
    nonzero = delta[delta != 0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    positives = int(np.sum(nonzero > 0))
    tail = sum(math.comb(n, k) for k in range(positives, n + 1)) / (2 ** n)
    return float(min(1.0, 2 * min(tail, 1 - tail + math.comb(n, positives) / (2 ** n))))


def _percent_change(base: float, delta: float) -> float | None:
    return None if base == 0 else (delta / base) * 100.0


def cross_check(left, right) -> Dict[str, object]:
    checks = []
    for method in METHODS:
        for source_query, target_query in zip(left["queries"], right["queries"]):
            terms = source_query["relevant_terms"]
            results = source_query["methods"][method]["results"]
            ids = [r["chunk_id"] for r in results]
            relevant = [
                r["chunk_id"]
                for r in results
                if sum(t.lower() in r["text"].lower() for t in terms) / max(len(terms) * 0.5, 1.0) >= 0.5
            ]
            manual_recall = len(set(ids[:10]) & set(relevant)) / len(set(relevant)) if relevant else 0.0
            manual_precision = len(set(ids[:10]) & set(relevant)) / 10
            metrics = source_query["methods"][method]["metrics"]
            checks.append(
                {
                    "collection": "384",
                    "method": method,
                    "query_id": source_query["id"],
                    "recall_match": abs(manual_recall - metrics["recall@10"]) < 1e-12,
                    "precision_match": abs(manual_precision - metrics["precision@10"]) < 1e-12,
                }
            )
    return {"all_match": all(c["recall_match"] and c["precision_match"] for c in checks), "checks": checks}


def compare_readme(result) -> Dict[str, object]:
    expected = {
        "bm25": {"recall@10": 0.8133, "precision@10": 0.2440, "mrr": 0.8400, "ndcg@10": 0.7208, "hit_rate@10": 1.0},
        "dense": {"recall@10": 0.8267, "precision@10": 0.2480, "mrr": 0.8667, "ndcg@10": 0.7516, "hit_rate@10": 1.0},
        "bm25_dense": {"recall@10": 0.9600, "precision@10": 0.2880, "mrr": 0.9600, "ndcg@10": 0.8631, "hit_rate@10": 1.0},
        "sparse": {"recall@10": 0.7733, "precision@10": 0.2320, "mrr": 0.8133, "ndcg@10": 0.6974, "hit_rate@10": 1.0},
        "sparse_dense": {"recall@10": 0.9600, "precision@10": 0.2880, "mrr": 0.9600, "ndcg@10": 0.8624, "hit_rate@10": 1.0},
        "rerank": {"recall@10": 0.9600, "precision@10": 0.2920, "mrr": 1.0000, "ndcg@10": 0.9312, "hit_rate@10": 1.0},
    }
    actual = {
        method: {
            metric: float(np.mean([q["methods"][method]["metrics"][metric] for q in result["queries"]]))
            for metric in expected[method]
        }
        for method in expected
    }
    return {"readme_expected": expected, "recreated_384": actual}


if __name__ == "__main__":
    main()
