"""Qdrant collection management and indexing pipeline."""

from __future__ import annotations

import hashlib
import time
from typing import List, Optional
from qdrant_client import QdrantClient, models
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from hybrid_search.config import AppConfig
from hybrid_search.embeddings.bm25 import BM25EmbeddingGenerator
from hybrid_search.embeddings.dense import DenseEmbeddingGenerator
from hybrid_search.embeddings.sparse import SpladeEmbeddingGenerator
from hybrid_search.models.document import Chunk


class QdrantIndexManager:
    """Manages Qdrant collection creation, payload indexing, and batch point ingestion."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        client: Optional[QdrantClient] = None,
        in_memory: bool = False,
    ):
        self.config = config or AppConfig()
        self.collection_name = self.config.qdrant.collection_name
        self.dense_vector_size = self.config.qdrant.dense_vector_size

        if client is not None:
            self.client = client
        elif in_memory:
            self.client = QdrantClient(":memory:")
        elif self.config.qdrant.prefer_local_embedded and self.config.qdrant.local_path:
            self.client = QdrantClient(path=self.config.qdrant.local_path)
        elif self.config.qdrant.url:
            try:
                self.client = QdrantClient(
                    url=self.config.qdrant.url,
                    api_key=self.config.qdrant.api_key,
                )
                # Ping client to verify connectivity
                self.client.get_collections()
            except Exception:
                # Graceful fallback to local embedded disk storage if remote server is not running
                local_path = self.config.qdrant.local_path or "qdrant_data"
                print(f"Notice: Qdrant server at {self.config.qdrant.url} unavailable. Using local embedded storage at './{local_path}'")
                self.client = QdrantClient(path=local_path)
        else:
            self.client = QdrantClient(":memory:")

    @classmethod
    def hash_to_int_id(cls, string_id: str) -> int:
        """Converts arbitrary string ID into a positive 63-bit integer for Qdrant point IDs."""
        return int(hashlib.md5(string_id.encode("utf-8")).hexdigest()[:15], 16)

    def create_collection(self, recreate: bool = False) -> None:
        """Creates the single hybrid search collection with 3 named vector spaces."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name in collections:
            if recreate:
                self.client.delete_collection(self.collection_name)
            else:
                return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=self.dense_vector_size,
                    distance=models.Distance.COSINE,
                )
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

        # Create payload indexes for metadata filtering
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
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception:
                pass

    def index_chunks(
        self,
        chunks: List[Chunk],
        dense_gen: Optional[DenseEmbeddingGenerator] = None,
        bm25_gen: Optional[BM25EmbeddingGenerator] = None,
        sparse_gen: Optional[SpladeEmbeddingGenerator] = None,
        batch_size: int = 100,
    ) -> int:
        """Generates embeddings and upserts chunks into Qdrant in streaming batches with real-time progress."""
        if not chunks:
            return 0

        self.create_collection(recreate=False)
        dense_gen = dense_gen or DenseEmbeddingGenerator(self.config.embeddings.dense_model)
        bm25_gen = bm25_gen or BM25EmbeddingGenerator(self.config.embeddings.bm25_model)
        sparse_gen = sparse_gen or SpladeEmbeddingGenerator(self.config.embeddings.sparse_model)

        total_chunks = len(chunks)
        total_batches = (total_chunks + batch_size - 1) // batch_size

        # Check existing indexed points for seamless resumption
        existing_point_count = 0
        try:
            col_info = self.client.get_collection(self.collection_name)
            existing_point_count = col_info.points_count or 0
        except Exception:
            pass

        if existing_point_count > 0:
            print(f"\n[STREAM-INDEXER] Resuming indexing ({existing_point_count:,} chunks already in Qdrant)...\n", flush=True)
        else:
            print(f"\n[STREAM-INDEXER] Starting multi-threaded indexing of {total_chunks:,} chunks across {total_batches} batches (batch size: {batch_size})...\n", flush=True)

        total_upserted = 0
        start_all = time.time()

        for b_idx in range(total_batches):
            start_i = b_idx * batch_size
            end_i = min(start_i + batch_size, total_chunks)

            # Skip batch if already fully indexed in Qdrant
            if existing_point_count >= end_i:
                total_upserted += (end_i - start_i)
                pct = (total_upserted / total_chunks) * 100.0
                print(f"[INDEXER] Batch {b_idx + 1:2d}/{total_batches} ({pct:5.1f}%): chunks {start_i + 1:4d}..{end_i:4d} [ALREADY INDEXED - SKIPPED]", flush=True)
                continue

            batch_chunks = chunks[start_i:end_i]
            batch_texts = [c.text for c in batch_chunks]

            t0 = time.time()
            # 1. Dense Embeddings
            dense_embs = dense_gen.embed_texts(batch_texts)

            # 2. BM25 Sparse Embeddings
            bm25_embs = bm25_gen.embed_texts(batch_texts)

            # 3. SPLADE Sparse Embeddings
            splade_embs = sparse_gen.embed_texts(batch_texts)
            t_emb = time.time() - t0


            points: List[models.PointStruct] = []
            for i, chunk in enumerate(batch_chunks):
                chunk.dense_vector = dense_embs[i]
                chunk.bm25_indices, chunk.bm25_values = bm25_embs[i]
                chunk.sparse_indices, chunk.sparse_values = splade_embs[i]

                point_id = self.hash_to_int_id(chunk.chunk_id)
                vector = {
                    "dense": chunk.dense_vector,
                    "bm25": models.SparseVector(
                        indices=chunk.bm25_indices,
                        values=chunk.bm25_values,
                    ),
                    "sparse": models.SparseVector(
                        indices=chunk.sparse_indices,
                        values=chunk.sparse_values,
                    ),
                }
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "company": chunk.company,
                    "ticker": chunk.ticker,
                    "cik": chunk.cik,
                    "document_type": chunk.document_type,
                    "filing_date": chunk.filing_date,
                    "fiscal_period": chunk.fiscal_period,
                    "section": chunk.section,
                    "sector": chunk.sector,
                    "geography": chunk.geography,
                    "text": chunk.text,
                    "text_hash": chunk.text_hash,
                    "chunk_index": chunk.chunk_index,
                    "source_url": chunk.source_url,
                    "accession_number": chunk.accession_number,
                }
                points.append(models.PointStruct(id=point_id, vector=vector, payload=payload))

            # Upsert current batch directly to Qdrant
            self.client.upsert(collection_name=self.collection_name, points=points)
            total_upserted += len(points)

            elapsed = time.time() - start_all
            avg_per_batch = elapsed / (b_idx + 1)
            remaining_batches = total_batches - (b_idx + 1)
            eta_mins = (remaining_batches * avg_per_batch) / 60.0

            pct = (total_upserted / total_chunks) * 100.0
            print(
                f"[INDEXER] Batch {b_idx + 1:2d}/{total_batches} ({pct:5.1f}%): "
                f"chunks {start_i + 1:4d}..{end_i:4d} indexed in {t_emb:4.1f}s | "
                f"Elapsed: {elapsed / 60:.1f}m | ETA: {eta_mins:.1f}m",
                flush=True,
            )

        total_time = time.time() - start_all
        print(f"\n[DONE] Successfully indexed {total_upserted:,} chunks into Qdrant in {total_time / 60:.2f} minutes.\n", flush=True)
        return total_upserted



