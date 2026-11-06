from __future__ import annotations

import json
import math
import time
from pathlib import Path

from qdrant_client import models

from common import (
    AUDIT,
    ROOT,
    SOURCE_COLLECTION,
    TARGET_COLLECTION,
    experiment_config,
    iter_points,
    open_local_client,
)
from hybrid_search.embeddings.dense import DenseEmbeddingGenerator


CHECKPOINT = AUDIT / "build_768_checkpoint.json"
MARKER = AUDIT / "build_768_started.json"


def main() -> None:
    AUDIT.mkdir(parents=True, exist_ok=True)
    client = open_local_client()
    collections = {item.name for item in client.get_collections().collections}
    if TARGET_COLLECTION in collections and not CHECKPOINT.exists():
        raise RuntimeError(
            f"{TARGET_COLLECTION} already exists without this experiment checkpoint; refusing to overwrite it"
        )
    source_info = client.get_collection(SOURCE_COLLECTION)
    source_count = int(source_info.points_count or 0)
    if TARGET_COLLECTION not in collections:
        client.create_collection(
            collection_name=TARGET_COLLECTION,
            vectors_config={
                "dense": models.VectorParams(size=768, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                    index=models.SparseIndexParams(on_disk=False),
                ),
                "sparse": models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                    index=models.SparseIndexParams(on_disk=False),
                ),
            },
        )
        for field, schema in [
            ("company", models.PayloadSchemaType.KEYWORD),
            ("ticker", models.PayloadSchemaType.KEYWORD),
            ("document_type", models.PayloadSchemaType.KEYWORD),
            ("filing_date", models.PayloadSchemaType.DATETIME),
            ("fiscal_period", models.PayloadSchemaType.KEYWORD),
            ("section", models.PayloadSchemaType.KEYWORD),
            ("sector", models.PayloadSchemaType.KEYWORD),
            ("geography", models.PayloadSchemaType.KEYWORD),
        ]:
            client.create_payload_index(
                collection_name=TARGET_COLLECTION,
                field_name=field,
                field_schema=schema,
            )
        MARKER.write_text(json.dumps({"source_count": source_count}), encoding="utf-8")

    checkpoint = {"processed_ids": [], "source_count": source_count}
    if CHECKPOINT.exists():
        checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    processed = set(checkpoint.get("processed_ids", []))
    generator = DenseEmbeddingGenerator(
        model_name="BAAI/bge-base-en-v1.5",
        batch_size=64,
    )
    started = time.perf_counter()
    batch = []
    total_seen = 0
    for point in iter_points(client, SOURCE_COLLECTION, with_vectors=True):
        total_seen += 1
        point_key = str(point.id)
        if point_key in processed:
            continue
        vector = point.vector or {}
        if not isinstance(vector, dict) or "bm25" not in vector or "sparse" not in vector:
            raise RuntimeError(f"Source point {point.id} lacks bm25/sparse vectors")
        batch.append(point)
        if len(batch) < 64:
            continue
        _upsert_batch(client, generator, batch, processed)
        batch = []
    if batch:
        _upsert_batch(client, generator, batch, processed)
    elapsed = time.perf_counter() - started
    target_count = int(client.get_collection(TARGET_COLLECTION).points_count or 0)
    result = {
        "source_count": source_count,
        "source_seen": total_seen,
        "target_count": target_count,
        "processed_count": len(processed),
        "dense_model": "BAAI/bge-base-en-v1.5",
        "embedding_seconds": elapsed,
    }
    (AUDIT / "build_768_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if target_count != source_count:
        raise RuntimeError(f"Target count {target_count} != source count {source_count}")


def _upsert_batch(client, generator, batch, processed) -> None:
    texts = [str(point.payload.get("text", "")) for point in batch]
    dense_vectors = generator.embed_texts(texts)
    if len(dense_vectors) != len(batch):
        raise RuntimeError("Embedding batch length mismatch")
    points = []
    for source, dense in zip(batch, dense_vectors):
        if len(dense) != 768 or not all(math.isfinite(value) for value in dense):
            raise RuntimeError(f"Invalid 768-dimensional vector for point {source.id}")
        source_vectors = source.vector
        points.append(
            models.PointStruct(
                id=source.id,
                vector={
                    "dense": dense,
                    "bm25": source_vectors["bm25"],
                    "sparse": source_vectors["sparse"],
                },
                payload=source.payload,
            )
        )
    client.upsert(collection_name=TARGET_COLLECTION, points=points, wait=True)
    processed.update(str(point.id) for point in batch)
    CHECKPOINT.write_text(
        json.dumps(
            {
                "processed_ids": sorted(processed),
                "processed_count": len(processed),
            }
        ),
        encoding="utf-8",
    )
    print(f"embedded_768={len(processed)}", flush=True)


if __name__ == "__main__":
    main()
