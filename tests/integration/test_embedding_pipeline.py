"""Integration tests for FastEmbed embedding generators."""

import pytest
from hybrid_search.embeddings.bm25 import BM25EmbeddingGenerator
from hybrid_search.embeddings.dense import DenseEmbeddingGenerator
from hybrid_search.embeddings.sparse import SpladeEmbeddingGenerator


@pytest.mark.integration
def test_dense_embedding_generation():
    dense_gen = DenseEmbeddingGenerator()
    texts = ["NVIDIA reported record Data Center revenue for fiscal year 2024."]
    embs = dense_gen.embed_texts(texts)
    assert len(embs) == 1
    assert len(embs[0]) == 384  # bge-small dimension

    query_emb = dense_gen.embed_query("NVIDIA data center revenue")
    assert len(query_emb) == 384


@pytest.mark.integration
def test_bm25_embedding_generation():
    bm25_gen = BM25EmbeddingGenerator()
    texts = ["Capital expenditure in AI accelerators grew across hyperscalers."]
    embs = bm25_gen.embed_texts(texts)
    assert len(embs) == 1
    indices, values = embs[0]
    assert len(indices) > 0
    assert len(indices) == len(values)

    q_indices, q_values = bm25_gen.embed_query("capital expenditure")
    assert len(q_indices) > 0
    assert len(q_indices) == len(q_values)


@pytest.mark.integration
def test_splade_embedding_generation():
    splade_gen = SpladeEmbeddingGenerator()
    texts = ["Supply chain constraints for high bandwidth memory chips."]
    embs = splade_gen.embed_texts(texts)
    assert len(embs) == 1
    indices, values = embs[0]
    assert len(indices) > 0
    assert len(indices) == len(values)
