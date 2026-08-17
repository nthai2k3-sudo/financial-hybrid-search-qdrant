"""Integration tests for Qdrant index manager."""

import pytest
from qdrant_client import QdrantClient
from hybrid_search.config import AppConfig
from hybrid_search.indexing.qdrant_index import QdrantIndexManager
from hybrid_search.models.document import Chunk


@pytest.mark.integration
def test_qdrant_collection_creation_and_indexing(sample_chunks: list[Chunk]):
    # Use in-memory Qdrant client for integration tests
    client = QdrantClient(":memory:")
    config = AppConfig()
    manager = QdrantIndexManager(config=config, client=client)

    manager.create_collection(recreate=True)
    collections = client.get_collections().collections
    assert any(c.name == config.qdrant.collection_name for c in collections)

    count = manager.index_chunks(sample_chunks)
    assert count == len(sample_chunks)

    # Check point count in collection
    info = client.get_collection(config.qdrant.collection_name)
    assert info.points_count == len(sample_chunks)
