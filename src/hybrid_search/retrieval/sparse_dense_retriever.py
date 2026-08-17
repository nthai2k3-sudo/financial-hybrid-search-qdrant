"""SPLADE + Dense hybrid retriever using Qdrant server-side RRF fusion."""

from __future__ import annotations

from typing import List, Optional
from qdrant_client import QdrantClient, models
from hybrid_search.embeddings.dense import DenseEmbeddingGenerator
from hybrid_search.embeddings.sparse import SpladeEmbeddingGenerator
from hybrid_search.filtering.metadata_filter import QdrantFilterBuilder
from hybrid_search.models.filter import MetadataFilter
from hybrid_search.models.search import SearchResult
from hybrid_search.retrieval.base import BaseRetriever


class SparseDenseRetriever(BaseRetriever):
    """Hybrid search combining SPLADE learned sparse vectors and Dense vectors with server-side RRF."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "financial_docs",
        dense_gen: Optional[DenseEmbeddingGenerator] = None,
        sparse_gen: Optional[SpladeEmbeddingGenerator] = None,
    ):
        super().__init__(client=client, collection_name=collection_name)
        self.dense_gen = dense_gen or DenseEmbeddingGenerator()
        self.sparse_gen = sparse_gen or SpladeEmbeddingGenerator()

    @property
    def name(self) -> str:
        return "sparse_dense"

    def search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[MetadataFilter] = None,
        prefetch_k: int = 50,
    ) -> List[SearchResult]:
        dense_vector = self.dense_gen.embed_query(query)
        sparse_indices, sparse_values = self.sparse_gen.embed_query(query)
        qdrant_filter = QdrantFilterBuilder.build_filter(filters)

        sparse_vector = models.SparseVector(indices=sparse_indices, values=sparse_values)

        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=prefetch_k,
                    filter=qdrant_filter,
                ),
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=prefetch_k,
                    filter=qdrant_filter,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k,
        )

        return self._points_to_search_results(response.points, self.name, query=query)
