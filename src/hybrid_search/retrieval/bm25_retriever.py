"""BM25 lexical retriever using Qdrant sparse vector."""

from __future__ import annotations

from typing import List, Optional
from qdrant_client import QdrantClient, models
from hybrid_search.embeddings.bm25 import BM25EmbeddingGenerator
from hybrid_search.filtering.metadata_filter import QdrantFilterBuilder
from hybrid_search.models.filter import MetadataFilter
from hybrid_search.models.search import SearchResult
from hybrid_search.retrieval.base import BaseRetriever


class BM25Retriever(BaseRetriever):
    """Lexical BM25 retrieval using Qdrant's 'bm25' named sparse vector space."""

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = "financial_docs",
        generator: Optional[BM25EmbeddingGenerator] = None,
    ):
        super().__init__(client=client, collection_name=collection_name)
        self.generator = generator or BM25EmbeddingGenerator()

    @property
    def name(self) -> str:
        return "bm25"

    def search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[MetadataFilter] = None,
        prefetch_k: int = 50,
    ) -> List[SearchResult]:
        q_indices, q_values = self.generator.embed_query(query)
        qdrant_filter = QdrantFilterBuilder.build_filter(filters)

        sparse_vector = models.SparseVector(indices=q_indices, values=q_values)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=sparse_vector,
            using="bm25",
            query_filter=qdrant_filter,
            limit=k,
        )

        return self._points_to_search_results(response.points, self.name, query=query)
