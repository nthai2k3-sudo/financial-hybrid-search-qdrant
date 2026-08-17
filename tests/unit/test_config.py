"""Unit tests for configuration loader."""

from hybrid_search.config import load_config


def test_load_default_config():
    config = load_config()
    assert config.edgar.rate_limit_per_second == 8.0
    assert config.chunking.chunk_size_tokens == 500
    assert config.embeddings.dense_model == "BAAI/bge-small-en-v1.5"
    assert config.embeddings.bm25_model == "Qdrant/bm25"
    assert config.embeddings.sparse_model == "prithivida/Splade_PP_en_v1"
    assert config.qdrant.collection_name == "financial_docs"
    assert len(config.companies) == 10

    nvda = next(c for c in config.companies if c.ticker == "NVDA")
    assert nvda.cik == "0001045810"
    assert "10-K" in nvda.forms
