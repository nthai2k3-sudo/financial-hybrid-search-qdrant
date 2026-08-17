"""Command line interface for the Financial Intelligence Hybrid Search system."""

from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from hybrid_search.config import load_config
from hybrid_search.chunking.chunker import DocumentChunker
from hybrid_search.evaluation.benchmark import BenchmarkRunner
from hybrid_search.evaluation.report import BenchmarkReporter
from hybrid_search.ingestion.pipeline import IngestionPipeline
from hybrid_search.indexing.qdrant_index import QdrantIndexManager
from hybrid_search.models.document import Document
from hybrid_search.models.filter import MetadataFilter
from hybrid_search.retrieval.engine import SearchEngine


console = Console()


def handle_ingest(args: argparse.Namespace) -> None:
    """Handles 'ingest' CLI command."""
    config = load_config()
    pipeline = IngestionPipeline(config=config)
    console.print(f"[bold cyan]Starting SEC EDGAR Ingestion Pipeline for {len(config.companies)} companies...[/bold cyan]")
    docs = pipeline.run(force_download=args.force)
    console.print(f"[bold green]Ingestion complete. Total documents processed: {len(docs)}[/bold green]")


def handle_index(args: argparse.Namespace) -> None:
    """Handles 'index' CLI command."""
    config = load_config()
    docs_path = Path("data/processed/documents.jsonl")
    if not docs_path.exists():
        console.print("[bold red]Error: data/processed/documents.jsonl not found. Run 'ingest' first.[/bold red]")
        sys.exit(1)

    documents = []
    with open(docs_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                documents.append(Document.model_validate_json(line))

    console.print(f"[bold cyan]Chunking {len(documents)} documents with contextual metadata...[/bold cyan]")
    chunker = DocumentChunker(
        chunk_size_tokens=config.chunking.chunk_size_tokens,
        chunk_overlap_tokens=config.chunking.chunk_overlap_tokens,
    )
    chunks_path = Path("data/processed/chunks.jsonl")
    chunks = chunker.chunk_documents(documents, output_path=chunks_path)
    console.print(f"[bold green]Generated {len(chunks)} contextualized chunks.[/bold green]")

    console.print(f"[bold cyan]Building Qdrant Index with 3 named vectors (dense, bm25, sparse)...[/bold cyan]")
    manager = QdrantIndexManager(config=config)
    manager.create_collection(recreate=args.recreate)
    indexed_count = manager.index_chunks(chunks)
    console.print(f"[bold green]Successfully indexed {indexed_count} points into Qdrant collection '{config.qdrant.collection_name}'.[/bold green]")


def handle_search(args: argparse.Namespace) -> None:
    """Handles 'search' CLI command."""
    config = load_config()
    engine = SearchEngine(config=config)

    # Build filter if provided
    metadata_filter: Optional[MetadataFilter] = None
    if args.ticker or args.form or args.section:
        metadata_filter = MetadataFilter(
            tickers=[args.ticker] if args.ticker else None,
            document_types=[args.form] if args.form else None,
            sections=[args.section] if args.section else None,
        )

    console.print(f"\n[bold yellow]Query:[/bold yellow] [bold white]{args.query}[/bold white]")
    if metadata_filter and not metadata_filter.is_empty():
        console.print(f"[dim]Filters: {metadata_filter.model_dump(exclude_none=True)}[/dim]")

    if args.method == "all":
        all_results = engine.search_all_configurations(
            query=args.query,
            k=args.k,
            filters=metadata_filter,
        )
        for method_name, results in all_results.items():
            _display_search_results(method_name, results, show_top_answer=args.verbose)
    else:
        results = engine.search(
            query=args.query,
            method=args.method,
            k=args.k,
            filters=metadata_filter,
        )
        _display_search_results(args.method, results, show_top_answer=True)


def _display_search_results(method: str, results: list, show_top_answer: bool = False) -> None:
    """Renders search results in a formatted table and highlights top factual excerpt."""
    table = Table(title=f"[bold cyan]Method: {method.upper()}[/bold cyan]", show_header=True)
    table.add_column("Rank", justify="center", style="cyan", width=6)
    table.add_column("Score", justify="right", style="green", width=10)
    table.add_column("Company", style="bold white", width=14)
    table.add_column("Form / Section", style="dim", width=26)
    table.add_column("Pinpointed Answer / Excerpt", style="white")

    for r in results:
        row_summary = r.exact_answer if r.exact_answer else ((r.text[:220] + "...") if len(r.text) > 220 else r.text)
        if len(row_summary) > 240:
            row_summary = row_summary[:240] + "..."
        table.add_row(
            str(r.rank),
            f"{r.score:.4f}",
            f"{r.ticker} ({r.company})",
            f"{r.document_type} | {r.section}",
            row_summary.replace("\n", " "),
        )

    console.print(table)

    if show_top_answer and results:
        top_match = results[0]

        # 1. Direct Extracted Answer Callout Box
        if top_match.exact_answer:
            console.print(
                Panel(
                    f"[bold yellow]\"{top_match.exact_answer}\"[/bold yellow]\n\n"
                    f"[dim]Source Filing:[/dim] [bold white]{top_match.company} ({top_match.ticker}) - {top_match.document_type} ({top_match.fiscal_period})[/bold white]\n"
                    f"[dim]Section:[/dim] [cyan]{top_match.section}[/cyan]  |  [dim]Score:[/dim] [green]{top_match.score:.4f}[/green]",
                    title=f"[bold green]>> DIRECT EXTRACTED ANSWER ({method.upper()}) <<[/bold green]",
                    border_style="green",
                )
            )

        # 2. Context Document Panel with Bold Highlight
        console.print(
            Panel(
                f"[bold green]Document Context:[/bold green] {top_match.company} ({top_match.ticker}) - {top_match.document_type} ({top_match.fiscal_period})\n"
                f"[bold green]Section:[/bold green] {top_match.section}\n\n"
                f"{top_match.highlighted_text or top_match.text}",
                title=f"[bold cyan]Full Document Context with Answering Clause ({method.upper()})[/bold cyan]",
                border_style="dim cyan",
            )
        )
    console.print()



def handle_benchmark(args: argparse.Namespace) -> None:
    """Handles 'benchmark' CLI command."""
    config = load_config()
    engine = SearchEngine(config=config)
    runner = BenchmarkRunner(engine=engine, config=config, dataset_path=args.dataset)

    results = runner.run_benchmark(
        k=args.k,
        num_latency_runs=args.runs,
        category_filter=args.category,
    )

    BenchmarkReporter.display_results_table(results)

    if args.output:
        out_path = Path(args.output)
        if out_path.suffix == ".json":
            out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        else:
            BenchmarkReporter.export_report_markdown(results, out_path)
        console.print(f"[bold green]Report exported to {out_path}[/bold green]")


def handle_info(args: argparse.Namespace) -> None:
    """Handles 'info' CLI command."""
    config = load_config()
    manager = QdrantIndexManager(config=config)
    try:
        col = manager.client.get_collection(config.qdrant.collection_name)
        console.print(f"[bold cyan]Collection:[/bold cyan] {config.qdrant.collection_name}")
        console.print(f"[bold cyan]Points Count:[/bold cyan] {col.points_count}")
        console.print(f"[bold cyan]Vectors Config:[/bold cyan] {col.config.params.vectors}")
        console.print(f"[bold cyan]Sparse Config:[/bold cyan] {col.config.params.sparse_vectors}")
    except Exception as err:
        console.print(f"[bold yellow]Collection status:[/bold yellow] Not found or inaccessible ({err})")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="hybrid-search",
        description="Financial Intelligence Hybrid Search CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # ingest
    ingest_p = subparsers.add_parser("ingest", help="Download and process SEC filings")
    ingest_p.add_argument("--force", action="store_true", help="Force re-download of existing filings")

    # index
    index_p = subparsers.add_parser("index", help="Chunk documents and build Qdrant index")
    index_p.add_argument("--recreate", action="store_true", help="Recreate Qdrant collection if exists")

    # search
    search_p = subparsers.add_parser("search", help="Query the financial search engine")
    search_p.add_argument("--query", "-q", required=True, help="Search query string")
    search_p.add_argument(
        "--method",
        "-m",
        default="sparse_dense",
        choices=["bm25", "dense", "bm25_dense", "sparse", "sparse_dense", "rerank", "all"],
        help="Retrieval configuration",
    )

    search_p.add_argument("--k", type=int, default=10, help="Number of results")
    search_p.add_argument("--ticker", help="Filter by ticker symbol (e.g. NVDA)")
    search_p.add_argument("--form", "--doc-type", dest="form", help="Filter by document type (e.g. 10-K, 10-Q)")
    search_p.add_argument("--section", help="Filter by section title")
    search_p.add_argument("--verbose", "-v", action="store_true", help="Show full top answer insight panel")


    # benchmark
    bench_p = subparsers.add_parser("benchmark", help="Run full evaluation and benchmarking suite")
    bench_p.add_argument("--k", type=int, default=10, help="Top-K evaluation rank")
    bench_p.add_argument("--runs", type=int, default=3, help="Latency measurement repetitions per query")
    bench_p.add_argument("--category", help="Run benchmark on a specific category only")
    bench_p.add_argument("--dataset", help="Custom golden dataset YAML path")
    bench_p.add_argument("--output", "-o", default="data/benchmark_report.md", help="Export path for report (.md or .json)")

    # info
    subparsers.add_parser("info", help="Display Qdrant collection and system information")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "ingest":
        handle_ingest(args)
    elif args.command == "index":
        handle_index(args)
    elif args.command == "search":
        handle_search(args)
    elif args.command == "benchmark":
        handle_benchmark(args)
    elif args.command == "info":
        handle_info(args)


if __name__ == "__main__":
    main()
