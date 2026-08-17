"""Dense vector semantic retriever using Qdrant."""

from __future__ import annotations

from typing import List, Optional
from qdrant_client import QdrantClient
from hybrid_search.embeddings.dense import DenseEmbeddingGenerator
from hybrid_search.filtering.metadata_filter import QdrantFilterBuilder
from hybrid_search.models.filter import MetadataFilter
from hybrid_search.models.search import SearchResult
from hybrid_search.retrieval.base import BaseRetriever


class DenseRetriever(BaseRetriever):
    """Semantic vector retrieval using Qdrant's 'dense' named vector space."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "financial_docs",
        generator: Optional[DenseEmbeddingGenerator] = None,
    ):
        super().__init__(client=client, collection_name=collection_name)
        self.generator = generator or DenseEmbeddingGenerator()

    @property
    def name(self) -> str:
        return "dense"

    def search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[MetadataFilter] = None,
        prefetch_k: int = 50,
    ) -> List[SearchResult]:
        query_vector = self.generator.embed_query(query)
        qdrant_filter = QdrantFilterBuilder.build_filter(filters)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using="dense",
            query_filter=qdrant_filter,
            limit=k,
        )

        return self._points_to_search_results(response.points, self.name, query=query)
