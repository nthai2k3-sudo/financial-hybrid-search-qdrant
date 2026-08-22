# Contributing to Financial Intelligence Hybrid Search

Thank you for your interest in contributing to the **Financial Intelligence Hybrid Search** project! We welcome contributions from engineers, researchers, and open-source contributors.

---

## Areas Where Contributions Are Welcomed

1. **New Embedding & Sparse Models:**
   - Adding support for alternative dense encoders (e.g., `Snowflake/snowflake-arctic-embed`, `nomic-ai/nomic-embed-text-v1.5`).
   - Integrating alternative sparse models (e.g., `naver/efficient-splade-VI-BT-large`).
2. **Vector Database Optimizations & Benchmarks:**
   - Testing Qdrant scalar and binary quantization (SQ/BQ) under financial retrieval accuracy constraints.
   - Expanding multi-vector prefetch topologies and evaluating alternative fusion algorithms (e.g., Relative Score Fusion / RSF vs RRF).
3. **Corpus Expansion & Data Ingestion:**
   - Adding support for other SEC filing types (e.g., Form 8-K material events, Form 13-F institutional holdings, S-1 IPO filings).
   - Ingesting filings for additional sectors (e.g., Financial Services, Healthcare, Energy).
4. **Benchmarking & Evaluation Datasets:**
   - Expanding `evaluation/golden_dataset.yaml` with more complex domain-specific financial queries.
   - Implementing automated synthetic query generation with LLM-as-a-judge relevance annotations.
5. **Web UI & Developer Experience:**
   - Enhancing visualization dashboards, search latency heatmaps, and table rendering components.

---

## Development Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/satyam671/financial-hybrid-search-qdrant.git
   cd financial-hybrid-search-qdrant
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate       # Windows PowerShell
   # source .venv/bin/activate    # Linux / macOS
   ```

3. **Install Dependencies in Editable Mode:**
   ```bash
   pip install -e .
   ```

4. **Verify Your Environment:**
   ```bash
   pytest tests/
   ```

---

## Contribution Workflow

1. **Create a Feature Branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Adhere to Code Standards:**
   - Use strict type annotations (`from __future__ import annotations`).
   - Maintain zero external paid API dependencies: All default models must run locally on CPU via FastEmbed / ONNX.
   - Follow Pydantic v2 conventions for schemas and configs.

3. **Run the Test Suite:**
   ```bash
   pytest tests/unit/ -v
   pytest tests/integration/ -v
   pytest tests/e2e/ -v
   ```

4. **Submit a Pull Request:**
   - Push your branch to GitHub.
   - Open a [Pull Request](https://github.com/satyam671/financial-hybrid-search-qdrant/pulls) against `main` with a clear description of the problem solved and test results.
   - For bug reports or feature ideas, please open an [Issue](https://github.com/satyam671/financial-hybrid-search-qdrant/issues).

---

## Code of Conduct

Please maintain a professional, collaborative, and inclusive environment.

