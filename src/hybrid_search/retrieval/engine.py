"""Unified search engine for executing single or multi-configuration searches."""

from __future__ import annotations

from typing import Dict, List, Optional
from qdrant_client import QdrantClient
from hybrid_search.config import AppConfig
from hybrid_search.embeddings.bm25 import BM25EmbeddingGenerator
from hybrid_search.embeddings.dense import DenseEmbeddingGenerator
from hybrid_search.embeddings.sparse import SpladeEmbeddingGenerator
from hybrid_search.indexing.qdrant_index import QdrantIndexManager
from hybrid_search.models.filter import MetadataFilter
from hybrid_search.models.search import SearchResult
from hybrid_search.retrieval.base import BaseRetriever
from hybrid_search.retrieval.bm25_dense_retriever import BM25DenseRetriever
from hybrid_search.retrieval.bm25_retriever import BM25Retriever
from hybrid_search.retrieval.dense_retriever import DenseRetriever
from hybrid_search.retrieval.reranker import NeuralReranker
from hybrid_search.retrieval.sparse_dense_retriever import SparseDenseRetriever
from hybrid_search.retrieval.sparse_retriever import SparseRetriever


class SearchEngine:
    """Unified engine encapsulating all retrieval configurations on Qdrant with Cross-Encoder Reranking."""

    METHOD_NAMES = ["bm25", "dense", "bm25_dense", "sparse", "sparse_dense", "rerank"]

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        client: Optional[QdrantClient] = None,
        in_memory: bool = False,
    ):
        self.config = config or AppConfig()
        self.index_manager = QdrantIndexManager(config=self.config, client=client, in_memory=in_memory)
        self.client = self.index_manager.client
        self.collection_name = self.config.qdrant.collection_name

        # Shared embedding model instances to avoid redundant loading
        self.dense_gen = DenseEmbeddingGenerator(self.config.embeddings.dense_model)
        self.bm25_gen = BM25EmbeddingGenerator(self.config.embeddings.bm25_model)
        self.sparse_gen = SpladeEmbeddingGenerator(self.config.embeddings.sparse_model)

        # Neural & Cross-Feature Reranker
        self.reranker = NeuralReranker()

        # Initialize base retrievers
        self.retrievers: Dict[str, BaseRetriever] = {
            "bm25": BM25Retriever(
                client=self.client,
                collection_name=self.collection_name,
                generator=self.bm25_gen,
            ),
            "dense": DenseRetriever(
                client=self.client,
                collection_name=self.collection_name,
                generator=self.dense_gen,
            ),
            "bm25_dense": BM25DenseRetriever(
                client=self.client,
                collection_name=self.collection_name,
                dense_gen=self.dense_gen,
                bm25_gen=self.bm25_gen,
            ),
            "sparse": SparseRetriever(
                client=self.client,
                collection_name=self.collection_name,
                generator=self.sparse_gen,
            ),
            "sparse_dense": SparseDenseRetriever(
                client=self.client,
                collection_name=self.collection_name,
                dense_gen=self.dense_gen,
                sparse_gen=self.sparse_gen,
            ),
        }

    def search(
        self,
        query: str,
        method: str = "sparse_dense",
        k: int = 10,
        filters: Optional[MetadataFilter] = None,
        prefetch_k: int = 50,
    ) -> List[SearchResult]:
        """Executes search with a specified retrieval configuration or neural reranking."""
        if method == "rerank":
            # Retrieve initial hybrid candidates from BM25+Dense and SPLADE+Dense
            base_candidates = self.retrievers["sparse_dense"].search(
                query=query,
                k=max(k * 3, 20),
                filters=filters,
                prefetch_k=prefetch_k,
            )
            # Rerank candidates with Cross-Feature Neural Scoring
            return self.reranker.rerank(query=query, results=base_candidates, top_k=k)

        if method not in self.retrievers:
            raise ValueError(f"Unknown retrieval method '{method}'. Valid options: {self.METHOD_NAMES}")

        retriever = self.retrievers[method]
        return retriever.search(query=query, k=k, filters=filters, prefetch_k=prefetch_k)

    def search_all_configurations(
        self,
        query: str,
        k: int = 10,
        filters: Optional[MetadataFilter] = None,
        prefetch_k: int = 50,
    ) -> Dict[str, List[SearchResult]]:
        """Executes the query across all retrieval configurations for side-by-side comparison."""
        results: Dict[str, List[SearchResult]] = {}
        for method_name, retriever in self.retrievers.items():
            results[method_name] = retriever.search(
                query=query,
                k=k,
                filters=filters,
                prefetch_k=prefetch_k,
            )

        # Add Neural Reranking over candidates
        results["rerank"] = self.search(
            query=query,
            method="rerank",
            k=k,
            filters=filters,
            prefetch_k=prefetch_k,
        )

        return results

