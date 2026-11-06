from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np

from common import (
    AUDIT,
    SOURCE_COLLECTION,
    TARGET_COLLECTION,
    dense_hash,
    experiment_config,
    iter_points,
    open_local_client,
    payload_hash,
    sparse_hash,
)
from hybrid_search.embeddings.dense import DenseEmbeddingGenerator


def main() -> None:
    client = open_local_client()
    source = list(iter_points(client, SOURCE_COLLECTION, with_vectors=True))
    target = list(iter_points(client, TARGET_COLLECTION, with_vectors=True))
    source_by_id = {str(point.id): point for point in source}
    target_by_id = {str(point.id): point for point in target}
    checks = {
        "counts_equal": len(source) == len(target),
        "ids_equal": set(source_by_id) == set(target_by_id),
        "payload_hash_equal": _hash_points(source_by_id, "payload")
        == _hash_points(target_by_id, "payload"),
        "bm25_hash_equal": _hash_points(source_by_id, "bm25")
        == _hash_points(target_by_id, "bm25"),
        "sparse_hash_equal": _hash_points(source_by_id, "sparse")
        == _hash_points(target_by_id, "sparse"),
    }
    dense_dimensions = []
    dense_finite = True
    dense_norms = []
    for point in target:
        dense = point.vector["dense"]
        dense_dimensions.append(len(dense))
        dense_finite = dense_finite and all(math.isfinite(float(value)) for value in dense)
        dense_norms.append(float(np.linalg.norm(np.asarray(dense, dtype=np.float64))))
    checks["dense_768_dimensions"] = all(dimension == 768 for dimension in dense_dimensions)
    checks["dense_finite"] = dense_finite
    checks["dense_l2_near_one"] = bool(
        dense_norms and all(abs(norm - 1.0) <= 1e-3 for norm in dense_norms)
    )

    rng = random.Random(384768)
    sample_ids = rng.sample(sorted(source_by_id), min(30, len(source_by_id)))
    generator = DenseEmbeddingGenerator(
        model_name="BAAI/bge-small-en-v1.5",
        batch_size=64,
    )
    texts = [str(source_by_id[point_id].payload["text"]) for point_id in sample_ids]
    recalculated = generator.embed_texts(texts)
    cosines = []
    for point_id, recalculated_vector in zip(sample_ids, recalculated):
        stored = np.asarray(source_by_id[point_id].vector["dense"], dtype=np.float64)
        fresh = np.asarray(recalculated_vector, dtype=np.float64)
        cosines.append(float(np.dot(stored, fresh) / (np.linalg.norm(stored) * np.linalg.norm(fresh))))
    checks["sample_384_cosine_near_one"] = bool(cosines and min(cosines) >= 0.999)
    result = {
        "checks": checks,
        "source_count": len(source),
        "target_count": len(target),
        "dense_norm_min": min(dense_norms),
        "dense_norm_max": max(dense_norms),
        "sample_size": len(sample_ids),
        "sample_cosine_min": min(cosines),
        "sample_cosine_max": max(cosines),
        "sample_cosines": dict(zip(sample_ids, cosines)),
    }
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT / "integrity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not all(checks.values()):
        raise RuntimeError("Integrity verification failed")


def _hash_points(points, kind: str) -> str:
    digest = hashlib.sha256()
    for point_id in sorted(points):
        point = points[point_id]
        if kind == "payload":
            value = payload_hash(point.payload)
        else:
            value = sparse_hash(point.vector[kind])
        digest.update(point_id.encode("utf-8"))
        digest.update(value.encode("ascii"))
    return digest.hexdigest()


if __name__ == "__main__":
    main()
