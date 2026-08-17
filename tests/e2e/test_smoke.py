"""End-to-end smoke test validating the complete system flow."""

import pytest
from qdrant_client import QdrantClient
from hybrid_search.config import AppConfig
from hybrid_search.chunking.chunker import DocumentChunker
from hybrid_search.evaluation.benchmark import BenchmarkRunner
from hybrid_search.models.document import Document, DocumentSection
from hybrid_search.retrieval.engine import SearchEngine


@pytest.mark.integration
def test_e2e_pipeline_smoke():
    # 1. Prepare sample synthetic documents
    doc1 = Document(
        document_id="0001045810_0001045810-24-000029",
        company="NVIDIA Corporation",
        ticker="NVDA",
        cik="0001045810",
        accession_number="0001045810-24-000029",
        document_type="10-K",
        filing_date="2024-02-21",
        report_date="2024-01-28",
        fiscal_period="FY2024",
        source_url="https://example.com/nvda",
        sector="Semiconductors",
        geography="US",
        sections=[
            DocumentSection(
                section_name="Item 1A: Risk Factors",
                section_code="ITEM_1A",
                text="NVIDIA faces manufacturing capacity constraints at TSMC for H100 and H200 accelerator platforms.",
            ),
            DocumentSection(
                section_name="Item 7: MD&A",
                section_code="ITEM_7",
                text="Capital expenditure in AI data center infrastructure expanded significantly across major hyperscalers.",
            ),
        ],
    )

    # 2. Chunk document
    chunker = DocumentChunker(chunk_size_tokens=100, chunk_overlap_tokens=20)
    chunks = chunker.chunk_documents([doc1])
    assert len(chunks) == 2

    # 3. Initialize engine with in-memory Qdrant
    client = QdrantClient(":memory:")
    config = AppConfig()
    engine = SearchEngine(config=config, client=client)

    # 4. Index chunks
    indexed_count = engine.index_manager.index_chunks(
        chunks,
        dense_gen=engine.dense_gen,
        bm25_gen=engine.bm25_gen,
        sparse_gen=engine.sparse_gen,
    )
    assert indexed_count == 2

    # 5. Search across all 5 configurations
    all_results = engine.search_all_configurations(
        query="NVIDIA H200 supply constraints",
        k=2,
    )
    for method in SearchEngine.METHOD_NAMES:
        assert method in all_results
        assert len(all_results[method]) > 0
        top = all_results[method][0]
        assert "H200" in top.text or "NVIDIA" in top.text

    # 6. Run benchmark runner
    runner = BenchmarkRunner(engine=engine, config=config)
    bench_results = runner.run_benchmark(k=2, num_latency_runs=1)
    assert "overall" in bench_results
    assert "latency" in bench_results
    for m in SearchEngine.METHOD_NAMES:
        assert m in bench_results["overall"]
        assert "ndcg@10" in bench_results["overall"][m]
