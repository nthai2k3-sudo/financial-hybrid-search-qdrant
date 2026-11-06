from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import yaml
from qdrant_client import QdrantClient, models

from hybrid_search.config import AppConfig, load_config
from hybrid_search.evaluation.metrics import IRMetrics

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = ROOT / "experiments" / "dim_384_vs_768"
AUDIT = EXPERIMENT / "audit"
RAW = EXPERIMENT / "results" / "raw"
SOURCE_COLLECTION = "financial_docs"
TARGET_COLLECTION = "financial_docs_bge768"
METHODS = ["bm25", "dense", "bm25_dense", "sparse", "sparse_dense", "rerank"]


def experiment_config(collection: str, dense_model: str) -> AppConfig:
    config = load_config(
        config_path=ROOT / "config" / "default.yaml",
        companies_path=ROOT / "config" / "companies.yaml",
    )
    qdrant = config.qdrant.model_copy(
        update={
            "collection_name": collection,
            "local_path": str(ROOT / "qdrant_data"),
            "prefer_local_embedded": True,
        }
    )
    embeddings = config.embeddings.model_copy(update={"dense_model": dense_model})
    return config.model_copy(update={"qdrant": qdrant, "embeddings": embeddings})


def open_local_client() -> QdrantClient:
    return QdrantClient(path=str(ROOT / "qdrant_data"))


def iter_points(client: QdrantClient, collection: str, with_vectors: bool = True):
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            offset=offset,
            limit=100,
            with_payload=True,
            with_vectors=with_vectors,
        )
        for point in points:
            yield point
        if offset is None:
            break


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def payload_hash(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def sparse_hash(vector: models.SparseVector) -> str:
    return hashlib.sha256(canonical_json({"indices": vector.indices, "values": vector.values})).hexdigest()


def dense_hash(vector: Sequence[float]) -> str:
    return hashlib.sha256(canonical_json([float(x) for x in vector])).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_queries() -> List[Dict[str, Any]]:
    with (ROOT / "evaluation" / "golden_dataset.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["queries"]


def relevance_score(chunk_text: str, relevant_terms: List[str]) -> float:
    text_lower = chunk_text.lower()
    if not relevant_terms:
        return 1.0
    matches = sum(1 for term in relevant_terms if term.lower() in text_lower)
    return min(matches / max(len(relevant_terms) * 0.5, 1.0), 1.0) if matches else 0.0


def metrics_for_results(results: List[Dict[str, Any]], relevant_terms: List[str]) -> Dict[str, float]:
    ids = [result["chunk_id"] for result in results]
    relevance = {
        result["chunk_id"]: relevance_score(result["text"], relevant_terms)
        for result in results
    }
    relevant_ids = [chunk_id for chunk_id, score in relevance.items() if score >= 0.5]
    return {
        "recall@10": IRMetrics.recall_at_k(ids, relevant_ids, 10),
        "precision@10": IRMetrics.precision_at_k(ids, relevant_ids, 10),
        "mrr": IRMetrics.reciprocal_rank(ids, relevant_ids),
        "ndcg@10": IRMetrics.ndcg_at_k(ids, relevance, 10),
        "hit_rate@10": IRMetrics.hit_rate_at_k(ids, relevant_ids, 10),
        **{
            f"recall@{k}": IRMetrics.recall_at_k(ids, relevant_ids, k)
            for k in [1, 3, 5, 20]
        },
    }


def system_metadata() -> Dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        commit = "unavailable"
    return {
        "timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "git_commit": commit,
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu_count": os.cpu_count(),
        "processor": platform.processor(),
    }
