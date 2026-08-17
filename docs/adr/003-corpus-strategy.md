# ADR-003: Financial Corpus Selection & Targeted Narrative Section Extraction

## Status
Accepted

## Context & Problem Statement
Financial search benchmarking requires real, authoritative corporate disclosures containing dense financial terminology, product names, accounting standards, and macroeconomic risk descriptions. 

Unstructured financial sources (earnings call transcripts, financial news blogs, broker research) have legal, licensing, or scraping barriers:
- Earnings call transcripts are proprietary copyright of providers (e.g. Motley Fool, Quartr) and lack public APIs.
- Financial news sites have aggressive anti-bot protections and volatile formatting.

Conversely, U.S. Securities and Exchange Commission (SEC) EDGAR filings (Forms 10-K, 10-Q, 20-F) are in the **public domain**, hosted on official RESTful endpoints (`data.sec.gov`), and represent the legal gold standard for financial intelligence. However, full annual 10-Ks are 50,000–100,000 words each. Ingesting ~120 full filings across 10 companies creates ~50,000 chunks, resulting in ~2 hours of CPU embedding time, which harms developer quickstart reproducibility.

## Considered Options

1. **Full Ingestion of All Filings (~120 filings, all sections, ~50,000 chunks):**
   - *Pros:* Complete filing representation.
   - *Cons:* Heavy boilerplate, hundreds of pages of raw financial tables that degrade text search; slow embedding generation on CPU (~2 hours).
2. **Synthetic / Mock Document Dataset:**
   - *Pros:* Tiny footprint, instant test runs.
   - *Cons:* Completely invalid for technical client evaluation and real IR benchmarking; lacks true domain complexity.
3. **Targeted SEC Narrative Ingestion (~50 filings, Items 1/1A/7/2, ~5,000–8,000 chunks):**
   - *Pros:* Extracts the richest textual narrative disclosures; reduces word count per filing from 75k to ~10k words; reduces total chunks to ~6,000; CPU embedding finishes in ~15–20 minutes; 100% public domain SEC data.
   - *Cons:* Financial statement tables are omitted.

## Decision Outcome
Adopt **Targeted SEC EDGAR Narrative Ingestion** across **10 key technology and semiconductor companies** for **FY2023–FY2025**:

### Target Companies Covered:
1. **NVIDIA (NVDA):** AI GPUs, H100/H200 supply chain constraints, Data Center revenue.
2. **Apple (AAPL):** Consumer tech, share repurchases, international supply chain.
3. **Microsoft (MSFT):** Azure cloud, OpenAI infrastructure capital deployment.
4. **Alphabet (GOOGL):** Google Cloud, TPU vs GPU infrastructure capex.
5. **Amazon (AMZN):** AWS cloud infrastructure, custom silicon (Trainium, Inferentia).
6. **Meta Platforms (META):** Generative AI server cluster deployment, capex guidance.
7. **Tesla (TSLA):** Manufacturing capex, energy storage, localized compute.
8. **AMD (AMD):** MI300 accelerator shipments, semiconductor competition.
9. **Intel (INTC):** Intel Foundry Services (IFS) operating losses, fab capacity buildout.
10. **TSMC (TSM):** 20-F / 6-K foreign filer, advanced packaging (CoWoS), Arizona fab expansion.

### Targeted Narrative Sections Extracted:
- **Item 1 / 4 (Business Overview):** Competitive landscape, product SKUs, market drivers.
- **Item 1A / 3.D (Risk Factors):** Geopolitical risks, supply bottlenecks, trade export controls.
- **Item 7 / 2 / 5 (MD&A):** Capital expenditures, revenue segment growth, liquidity.

## Data Provenance & Legal Compliance
- **Source:** Direct downloads from `https://www.sec.gov/Archives/edgar/`.
- **License:** Works of the U.S. Federal Government / Public Domain under U.S. copyright law.
- **Fair Access Policy:** Ingestion adheres strictly to SEC rate limits (token bucket max 8 req/sec) with declarative `User-Agent` identification.
