"""Comprehensive benchmark runner evaluating all 5 retrieval configurations."""

from __future__ import annotations

import time
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from hybrid_search.config import AppConfig, get_project_root
from hybrid_search.evaluation.latency import LatencyStats, LatencyTracker
from hybrid_search.evaluation.metrics import IRMetrics
from hybrid_search.retrieval.engine import SearchEngine


class BenchmarkRunner:
    """Executes systematic benchmarks evaluating retrieval quality and latency across 5 configurations."""

    def __init__(
        self,
        engine: SearchEngine,
        config: Optional[AppConfig] = None,
        dataset_path: Optional[str | Path] = None,
    ):
        self.engine = engine
        self.config = config or AppConfig()
        root = get_project_root()
        self.dataset_path = Path(dataset_path) if dataset_path else root / "evaluation" / "golden_dataset.yaml"

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Loads evaluation queries from YAML."""
        if not self.dataset_path.exists():
            return []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data.get("queries", [])

    def score_query_relevance(self, chunk_text: str, relevant_terms: List[str]) -> float:
        """Determines binary or graded relevance score based on query semantic term coverage."""
        text_lower = chunk_text.lower()
        if not relevant_terms:
            return 1.0

        matches = sum(1 for term in relevant_terms if term.lower() in text_lower)
        if matches == 0:
            return 0.0
        return min(matches / max(len(relevant_terms) * 0.5, 1.0), 1.0)

    def run_benchmark(
        self,
        k: int = 10,
        num_latency_runs: int = 3,
        category_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs evaluation over the benchmark dataset."""
        queries = self.load_dataset()
        if category_filter:
            queries = [q for q in queries if q.get("category") == category_filter]

        methods = SearchEngine.METHOD_NAMES
        latency_trackers: Dict[str, LatencyTracker] = {m: LatencyTracker() for m in methods}

        # Store metric samples: {method: {metric: [values]}}
        method_metrics: Dict[str, Dict[str, List[float]]] = {
            m: {
                "recall@10": [],
                "precision@10": [],
                "mrr": [],
                "ndcg@10": [],
                "hit_rate@10": [],
            }
            for m in methods
        }

        category_metrics: Dict[str, Dict[str, Dict[str, List[float]]]] = {}

        print(f"Running benchmark across {len(queries)} queries and 5 retrieval configurations...")

        for q_idx, q_item in enumerate(queries, start=1):
            query_text = q_item["query"]
            cat = q_item.get("category", "general")
            relevant_terms = q_item.get("relevant_terms", [])

            if cat not in category_metrics:
                category_metrics[cat] = {
                    m: {
                        "recall@10": [],
                        "precision@10": [],
                        "mrr": [],
                        "ndcg@10": [],
                        "hit_rate@10": [],
                    }
                    for m in methods
                }

            # Warmup and execute across methods
            for method in methods:
                # Latency runs
                latencies: List[float] = []
                last_results = []
                for _ in range(num_latency_runs):
                    t0 = time.perf_counter()
                    results = self.engine.search(query=query_text, method=method, k=k)
                    t1 = time.perf_counter()
                    latencies.append((t1 - t0) * 1000.0)  # ms
                    last_results = results

                # Record latency
                for lat in latencies:
                    latency_trackers[method].record("total", lat)


                # Compute relevance
                retrieved_ids = [r.chunk_id for r in last_results]
                relevance_map = {}
                relevant_ids = []
                for r in last_results:
                    rel_score = self.score_query_relevance(r.text, relevant_terms)
                    relevance_map[r.chunk_id] = rel_score
                    if rel_score >= 0.5:
                        relevant_ids.append(r.chunk_id)

                # Calculate IR metrics
                rec = IRMetrics.recall_at_k(retrieved_ids, relevant_ids, k=k)
                prec = IRMetrics.precision_at_k(retrieved_ids, relevant_ids, k=k)
                mrr = IRMetrics.reciprocal_rank(retrieved_ids, relevant_ids)
                hit = IRMetrics.hit_rate_at_k(retrieved_ids, relevant_ids, k=k)
                ndcg = IRMetrics.ndcg_at_k(retrieved_ids, relevance_map, k=k)

                # Record
                for metric_name, val in [
                    ("recall@10", rec),
                    ("precision@10", prec),
                    ("mrr", mrr),
                    ("ndcg@10", ndcg),
                    ("hit_rate@10", hit),
                ]:
                    method_metrics[method][metric_name].append(val)
                    category_metrics[cat][method][metric_name].append(val)

        # Aggregate overall summary
        overall_summary: Dict[str, Dict[str, float]] = {}
        for m in methods:
            overall_summary[m] = {
                metric: float(np.mean(vals)) if vals else 0.0
                for metric, vals in method_metrics[m].items()
            }

        # Aggregate latency stats
        latency_summary: Dict[str, Dict[str, float]] = {}
        for m in methods:
            stats = latency_trackers[m].get_stats("total")
            latency_summary[m] = stats.model_dump()

        # Aggregate category summary
        category_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
        for cat, cat_data in category_metrics.items():
            category_summary[cat] = {}
            for m in methods:
                category_summary[cat][m] = {
                    metric: float(np.mean(vals)) if vals else 0.0
                    for metric, vals in cat_data[m].items()
                }

        results_payload = {
            "total_queries": len(queries),
            "k": k,
            "overall": overall_summary,
            "latency": latency_summary,
            "by_category": category_summary,
        }

        return results_payload
