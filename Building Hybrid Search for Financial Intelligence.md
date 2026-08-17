# Building Hybrid Search for Financial Intelligence

**Developer-in-Charge:** Satyam Sahu

> A practical guide to building hybrid search, starting with traditional
> BM25 + dense vectors and then comparing it with sparse-vector-based
> hybrid search. Include architecture, retrieval quality, latency, and
> implementation examples.

------------------------------------------------------------------------

## 1. Introduction: An Analyst's Search Problem

### 1.1 Opening Financial Research Query

Open with a realistic financial research query:

> "Find evidence that semiconductor companies are reducing capital
> expenditure because of weaker AI accelerator demand."

A semantic search system might retrieve documents discussing:

-   slowing infrastructure investment
-   weaker data-center spending
-   reduced expansion plans
-   declining accelerator demand

### 1.2 Exact-Term Financial Research Query

But now consider a different query:

> "Find references to NVIDIA's H200 supply constraints in Q4 filings."

Here, exact terms matter enormously:

-   NVIDIA
-   H200
-   supply constraints
-   Q4
-   specific filings

### 1.3 Central Problem

Introduce the central problem:

**Financial search requires both understanding meaning and matching
exact language.**

This is the fundamental motivation for hybrid search.

------------------------------------------------------------------------

## 2. Why Dense Vector Search Alone Isn't Enough

### 2.1 What Dense Retrieval Is Good At

Explain what dense retrieval is good at.

For example:

> "Companies are cutting infrastructure investment because AI demand is
> cooling."

A dense embedding can retrieve a document saying:

> "Management expects data-center expansion to moderate as accelerator
> demand normalizes."

The wording is different, but the meaning is similar.

### 2.2 Where Dense Retrieval Struggles

However, dense retrieval can struggle with:

-   company names
-   ticker symbols
-   product names
-   legal terminology
-   technical identifiers
-   specific accounting terms
-   exact phrases
-   numbers and codes

Give examples:

-   Query: "H200"
-   Query: "ASC 606"
-   Query: "NVDA"
-   Query: "10-K"
-   Query: "gross margin"

These aren't necessarily semantic concepts. They're often **precise
lexical identifiers**.

------------------------------------------------------------------------

## 3. BM25: The Classical Approach

### 3.1 What BM25 Does Well

Introduce BM25 as the classic lexical retrieval approach.

Explain its strengths:

-   exact terms
-   term frequency
-   document frequency
-   rare keywords
-   phrase-like matching
-   highly specific terminology

Use examples from financial documents.

### 3.2 Example: Exact-Term Retrieval

Query:

> "share repurchase authorization"

BM25 can strongly favor documents containing those exact terms.

Explain why this can outperform dense retrieval for certain searches.

### 3.3 Limitation of Lexical Retrieval

But then demonstrate the limitation.

Query:

> "companies reducing investment in new factories"

A document saying:

> "Management expects lower capital deployment toward capacity
> expansion"

may be highly relevant but contain none of the query's exact language.

This establishes the complementary nature of lexical and semantic
search.

------------------------------------------------------------------------

## 4. The Hybrid Search Idea

### 4.1 Basic Architecture

Introduce the basic architecture:

``` text
                   User Query
                       │
              ┌─────────┴─────────┐
              ↓                   ↓
         BM25 / Lexical      Dense Vector
              │                   │
              ↓                   ↓
        Top-K results        Top-K results
              │                   │
              └─────────┬─────────┘
                        ↓
                   Fusion / Ranking
                        ↓
                   Final results
```

### 4.2 Core Principle

Explain the fundamental principle:

**BM25 finds documents that use the right words.**

**Dense vectors find documents that express the right ideas.**

Hybrid search combines both.

------------------------------------------------------------------------

## 5. A Real Financial Research Dataset

### 5.1 Financial Document Corpus

Define a realistic corpus:

-   SEC filings
-   annual reports
-   quarterly filings
-   earnings-call transcripts
-   investor presentations
-   financial news
-   company reports
-   analyst research

### 5.2 Document and Chunk Fields

Each document/chunk could contain:

-   company
-   ticker
-   document_type
-   filing_date
-   fiscal_period
-   section
-   text
-   dense_vector
-   sparse_vector

### 5.3 Why Financial Corpora Are Challenging

Explain why financial corpora are particularly challenging:

-   highly repetitive terminology
-   company-specific vocabulary
-   abbreviations
-   numbers
-   named entities
-   rapidly changing terminology
-   domain-specific concepts

------------------------------------------------------------------------

## 6. Building the First Hybrid System: BM25 + Dense Vectors

### 6.1 Initial Implementation Approach

Start with the familiar approach.

For every document:

1.  Generate a dense embedding.
2.  Store the text for lexical retrieval.
3.  Run BM25 for keyword search.
4.  Run dense vector search for semantic retrieval.
5.  Combine the results.

### 6.2 Architecture and Data Flow

Explain the architecture and data flow.

### 6.3 Qdrant-Oriented Implementation

Show a Qdrant-oriented implementation example.

Explain why you as a developer have chosen Qdrant.

Qdrant website: https://qdrant.tech/

Keep the code practical rather than merely showing pseudocode.

------------------------------------------------------------------------

## 7. The Retrieval Quality Problem: Two Search Systems, Two Rankings

### 7.1 Different Rankings From Different Retrieval Systems

Suppose BM25 returns:

1.  Document A
2.  Document B
3.  Document C

while dense retrieval returns:

1.  Document D
2.  Document B
3.  Document F

The systems produce different rankings.

So how do we combine them?

### 7.2 Rank Fusion

Introduce **rank fusion**.

Discuss approaches such as:

-   weighted score combination
-   Reciprocal Rank Fusion (RRF)
-   rank-based fusion
-   score normalization

Explain why simply adding BM25 and vector scores can be problematic
because the scores may live on different scales.

------------------------------------------------------------------------

## 8. Reciprocal Rank Fusion: A Practical Hybrid Strategy

### 8.1 RRF Intuition

Use RRF as the concrete example.

Explain the intuition:

A document appearing near the top of multiple retrieval systems should
receive a strong combined ranking.

### 8.2 RRF Example

BM25:

-   A → rank 1
-   B → rank 2
-   C → rank 3

Dense:

-   D → rank 1
-   B → rank 2
-   F → rank 3

Document B becomes interesting because **both retrieval systems
independently consider it highly relevant**.

### 8.3 Why Use RRF as a Starting Point

Explain why rank fusion can be an effective starting point without
requiring extensive score calibration.

Then demonstrate the approach using Qdrant.

------------------------------------------------------------------------

## 9. Enter Sparse Vectors

### 9.1 Sparse + Dense as an Alternative Architecture

Now introduce the second half of the article.

Instead of maintaining:

**BM25 + dense vectors**

we can use:

**Sparse vectors + dense vectors**

### 9.2 What Sparse Vectors Represent

Explain what sparse vectors represent.

Unlike dense embeddings:

``` text
dense:
[0.13, -0.42, 0.08, ...]
```

a sparse representation has only a relatively small number of non-zero
dimensions:

``` text
token/term → weight
```

This allows the representation to retain much of the **lexical
specificity** that traditional keyword search provides while operating
within a vector-search architecture.

### 9.3 Learned Sparse Retrieval

Introduce modern learned sparse retrieval models such as SPLADE as an
example.

------------------------------------------------------------------------

## 10. Why Sparse Vectors Can Be Better Than Traditional BM25

### 10.1 Conceptual Difference

Explain the key conceptual difference.

BM25 is based on lexical statistics.

Learned sparse retrieval can learn which terms are important for
retrieval.

### 10.2 Learned Relationships Between Terms

This allows sparse representations to capture relationships such as:

-   "capital expenditure" → "capex"
-   "artificial intelligence accelerator" → potentially related learned
    terms
-   "share repurchase" → related terminology

The exact capabilities depend on the sparse model.

### 10.3 Positioning Sparse Retrieval

Emphasize that sparse retrieval sits somewhere between:

**traditional lexical search**

and

**dense semantic search**

while retaining a highly interpretable sparse representation.

------------------------------------------------------------------------

## 11. BM25 vs Sparse Vectors: What Actually Changes?

Create a direct comparison.

| Feature | BM25 | Sparse Vectors | Dense Vectors |
| :--- | :--- | :--- | :--- |
| **Exact terms** | Excellent | Excellent | Variable |
| **Semantic similarity** | Limited | Better | Excellent |
| **Rare terminology** | Excellent | Excellent | Variable |
| **Domain adaptation** | Limited | Model-dependent | Model-dependent |
| **Interpretability** | High | Relatively high | Low |
| **Infrastructure** | Traditional search | Vector DB | Vector DB |
| **Hybrid potential** | Excellent | Excellent | Excellent |

Discuss the important point:

**Sparse vectors don't simply replace dense vectors. They provide
another retrieval signal.**

The most interesting architecture remains:

**Sparse + Dense**

------------------------------------------------------------------------

## 12. Implementing Sparse + Dense Hybrid Search with Qdrant

### 12.1 Collection Design

Move into the implementation.

Show how the collection can contain:

-   dense vector
-   sparse vector
-   payload metadata

### 12.2 Retrieval Architecture

Then demonstrate:

``` text
User query
    ↓
Dense embedding
+
Sparse representation
    ↓
Qdrant
    ↓
Dense retrieval + sparse retrieval
    ↓
Fusion
    ↓
Top results
```

### 12.3 Concrete Qdrant Examples

Include concrete Qdrant examples for:

-   creating the collection
-   defining dense and sparse vectors
-   inserting documents
-   generating query representations
-   performing hybrid search

The code should be runnable or close to runnable rather than conceptual
pseudo-code.

------------------------------------------------------------------------

## 13. A Real Query: Exact Entity vs Semantic Concept

Use several examples to show when each retrieval method wins.

### 13.1 Query 1 --- Exact Terminology

> "NVDA H200 supply constraints"

Likely strength:

**BM25 / sparse retrieval**

Because exact product/company terminology matters.

### 13.2 Query 2 --- Conceptual Question

> "Which companies are seeing weaker demand for AI infrastructure?"

Likely strength:

**Dense retrieval**

because relevant documents may use very different terminology.

### 13.3 Query 3 --- Mixed Query

> "How are hyperscaler capex plans changing because of slowing AI
> accelerator demand?"

Potentially strong for:

**Hybrid retrieval**

because it combines:

-   specific concepts
-   domain terminology
-   potentially exact terms

This section should make the trade-off tangible.

------------------------------------------------------------------------

## 14. Latency: What Does Hybrid Search Cost?

Hybrid retrieval isn't free.

### 14.1 Dense-Only Retrieval

``` text
Query
 ↓
Embedding
 ↓
Vector search
```

### 14.2 BM25 + Dense Retrieval

``` text
Query
 ├── BM25
 └── Dense
       ↓
      Fusion
```

### 14.3 Sparse + Dense Retrieval

``` text
Query
 ├── Sparse encoding
 └── Dense encoding
       ↓
      Hybrid retrieval
```

### 14.4 Latency and Infrastructure Considerations

Discuss:

-   query encoding cost
-   retrieval latency
-   number of searches
-   fusion overhead
-   indexing cost
-   memory requirements

Explain where hybrid retrieval adds latency and where it can remain
practical for production systems.

------------------------------------------------------------------------

## 15. Retrieval Quality: How Should We Actually Measure It?

### 15.1 Evaluation Metrics

Don't rely on subjective examples.

Introduce evaluation metrics:

-   Recall@K
-   Precision@K
-   MRR
-   NDCG
-   hit rate
-   latency

### 15.2 Evaluation Dataset

Build a small evaluation dataset containing financial queries with known
relevant documents.

For each query, compare:

1.  BM25
2.  Dense retrieval
3.  BM25 + dense
4.  Sparse retrieval
5.  Sparse + dense

This creates a proper empirical comparison rather than an opinion piece.

------------------------------------------------------------------------

## 16. Designing the Benchmark

### 16.1 Category A --- Exact Terminology

Company names, tickers, product names, accounting standards.

### 16.2 Category B --- Semantic Questions

Conceptual queries where the answer may be expressed using different
terminology.

### 16.3 Category C --- Mixed Queries

Queries combining specific entities with conceptual intent.

### 16.4 Category D --- Rare Terminology

Highly domain-specific terms that appear infrequently.

### 16.5 Category E --- Long Natural-Language Questions

Questions written the way an analyst might actually ask an AI research
system.

### 16.6 Benchmark Comparison

Then compare retrieval quality and latency across approaches.

------------------------------------------------------------------------

## 17. Where BM25 Still Wins

Avoid presenting sparse vectors as an automatic replacement.

Discuss situations where BM25 can remain extremely effective:

-   exact identifiers
-   highly constrained keyword searches
-   known terminology
-   simple corpora
-   existing Elasticsearch/OpenSearch infrastructure
-   environments where predictable lexical behavior matters

This makes the article more credible.

------------------------------------------------------------------------

## 18. Where Sparse Vectors Become Interesting

Discuss scenarios where learned sparse retrieval offers advantages:

-   domain-specific vocabulary
-   semantic expansion
-   combining lexical and semantic signals
-   vector-native infrastructure
-   hybrid retrieval at scale

Explain that sparse retrieval can provide a more flexible lexical signal
than traditional BM25 while still being fundamentally different from
dense embeddings.

------------------------------------------------------------------------

## 19. The Production Architecture

Bring everything together.

A production financial research system might look like:

``` text
                   User
                     │
                     ↓
                 Query parser
                     │
           ┌──────────┴──────────┐
           ↓                     ↓
      Dense encoder        Sparse encoder
           │                     │
           └──────────┬──────────┘
                      ↓
                    Qdrant
                      │
                 Hybrid retrieval
                      ↓
                   Reranking
                      ↓
                    LLM / RAG
                      ↓
                 Cited research answer
```

### 19.1 Metadata Filtering

Explain where metadata filtering fits into this architecture:

-   company
-   date
-   document type
-   fiscal period
-   geography
-   sector

This connects hybrid search with the previous article's theme of
**advanced filtering** without making this article dependent on it.

------------------------------------------------------------------------

## 20. Conclusion: Search Isn't One Retrieval Problem

End by challenging the assumption that there is a single "best"
retrieval method.

Different signals answer different questions:

### BM25

> "Does this document contain the words I'm looking for?"

### Sparse Retrieval

> "Does this document contain or strongly represent the important
> lexical concepts?"

### Dense Retrieval

> "Does this document express the same meaning?"

Hybrid search asks:

> **"Can we use all three signals to build a better retrieval system?"**

Conclude that for serious RAG systems, the interesting engineering
problem isn't choosing between keyword search and vector search.

It is designing a retrieval architecture where **lexical precision and
semantic recall complement each other**.

------------------------------------------------------------------------

## 21. Technical Requirements for the Writer

-   Use **financial research** consistently as the primary domain.
-   The article should be a practical technical guide, not a theoretical
    explanation.
-   Include real Qdrant implementation examples for:
    -   dense vectors
    -   sparse vectors
    -   hybrid queries
    -   fusion
    -   metadata filtering
-   Clearly explain the difference between:
    -   BM25
    -   traditional sparse/lexical retrieval
    -   learned sparse vectors
    -   dense vectors
-   Include a **BM25 + dense vs sparse + dense** comparison.
-   Include a realistic benchmark methodology covering:
    -   retrieval quality
    -   recall
    -   ranking quality
    -   latency
    -   indexing/storage considerations
-   Use concrete financial-search queries throughout.
-   Include architecture diagrams.
-   Avoid claiming that sparse vectors universally outperform BM25. The
    article should demonstrate **where each approach works best**.
-   Keep Qdrant central to the implementation, but make the article
    useful to an engineer even if they are evaluating multiple vector
    databases.

------------------------------------------------------------------------

## 22. Final Production Notes

-   Remember to add 2-3 illustrations / diagrams.
-   End with a References section.
