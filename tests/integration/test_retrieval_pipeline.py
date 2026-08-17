"""Integration tests for all 5 Qdrant retrieval configurations."""

import pytest
from qdrant_client import QdrantClient
from hybrid_search.config import AppConfig
from hybrid_search.models.document import Chunk
from hybrid_search.models.filter import MetadataFilter
from hybrid_search.retrieval.engine import SearchEngine


@pytest.fixture
def initialized_search_engine(sample_chunks: list[Chunk]) -> SearchEngine:
    client = QdrantClient(":memory:")
    config = AppConfig()
    engine = SearchEngine(config=config, client=client)

    # Index sample chunks
    engine.index_manager.index_chunks(
        sample_chunks,
        dense_gen=engine.dense_gen,
        bm25_gen=engine.bm25_gen,
        sparse_gen=engine.sparse_gen,
    )
    return engine


@pytest.mark.integration
@pytest.mark.parametrize("method", ["bm25", "dense", "bm25_dense", "sparse", "sparse_dense", "rerank"])
def test_all_retrieval_configurations(initialized_search_engine: SearchEngine, method: str):
    results = initialized_search_engine.search(
        query="NVIDIA supply constraints H200 accelerators",
        method=method,
        k=2,
    )
    assert len(results) > 0
    top = results[0]
    assert top.retrieval_method == method
    assert top.rank == 1
    assert top.chunk_id != ""
    assert top.text != ""


@pytest.mark.integration
def test_metadata_filtered_search(initialized_search_engine: SearchEngine):
    # Filter only AAPL documents
    results = initialized_search_engine.search(
        query="capital expenditure repurchase",
        method="sparse_dense",
        k=5,
        filters=MetadataFilter(tickers=["AAPL"]),
    )
    assert len(results) > 0
    for r in results:
        assert r.ticker == "AAPL"


@pytest.mark.integration
def test_search_all_configurations(initialized_search_engine: SearchEngine):
    all_results = initialized_search_engine.search_all_configurations(
        query="capital expenditure AI accelerators",
        k=2,
    )
    assert len(all_results) == len(SearchEngine.METHOD_NAMES)
    for method in SearchEngine.METHOD_NAMES:
        assert method in all_results
        assert len(all_results[method]) > 0
