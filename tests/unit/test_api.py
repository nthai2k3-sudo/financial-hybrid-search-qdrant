"""Unit tests for FastAPI REST API endpoints."""

from fastapi.testclient import TestClient
from hybrid_search.api import create_app
from hybrid_search.retrieval.engine import SearchEngine


def test_api_status_and_metadata():
    engine = SearchEngine(in_memory=True)
    app = create_app(engine=engine)
    client = TestClient(app)

    # Test /api/status
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert data["collection"] == "financial_docs"
    assert "dense_model" in data

    # Test /api/metadata
    res_meta = client.get("/api/metadata")
    assert res_meta.status_code == 200
    meta_data = res_meta.json()
    assert len(meta_data["tickers"]) > 0
    assert "10-K" in meta_data["form_types"]
    assert len(meta_data["methods"]) == 6


def test_api_index_html_serving():
    engine = SearchEngine(in_memory=True)
    app = create_app(engine=engine)
    client = TestClient(app)
    res = client.get("/")
    assert res.status_code == 200
    assert "Financial Intelligence Hybrid Search" in res.text


def test_api_search_endpoint():
    engine = SearchEngine(in_memory=True)
    engine.index_manager.create_collection(recreate=True)
    app = create_app(engine=engine)
    client = TestClient(app)

    payload = {
        "query": "revenue recognition accounting policy",
        "method": "bm25",
        "tickers": ["TSLA"],
        "form_types": ["10-K"],
        "k": 3
    }
    res = client.post("/api/search", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["query"] == payload["query"]
    assert data["method"] == "bm25"
    assert "latency_ms" in data
    assert "results" in data
