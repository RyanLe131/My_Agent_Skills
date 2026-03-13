# Context Retrieval

Query the vector store to find the most relevant chunks for a given user question.

## Retrieval Strategies

| Strategy | How It Works | Best For |
|----------|-------------|----------|
| Dense (semantic) | Embed query → nearest neighbor search | Conceptual similarity |
| Sparse (BM25/TF-IDF) | Keyword matching with term frequency | Exact keyword matches |
| Hybrid | Combine dense + sparse scores | Best of both worlds |
| Multi-query | Generate query variants → merge results | Ambiguous questions |
| HyDE | Generate hypothetical answer → embed that | Bridging query-doc gap |
| Step-back | Abstract the question → retrieve broader context | Complex reasoning |
| Parent-child | Retrieve small chunks → return parent context | Precision + full context |

## Basic Retrieval

```python
# LangChain
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4},
)
docs = retriever.invoke("What is the refund policy?")

# With similarity score threshold
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7, "k": 10},
)

# Maximal Marginal Relevance (diverse results)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5},
)
```

```python
# LlamaIndex
query_engine = index.as_query_engine(similarity_top_k=4)
retriever = index.as_retriever(similarity_top_k=4)
nodes = retriever.retrieve("What is the refund policy?")
```

## Metadata Filtering

```python
# LangChain — filter by metadata before similarity search
retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4,
        "filter": {"category": "legal", "year": 2024},
    }
)

# Qdrant advanced filtering
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range

filter_ = Filter(
    must=[
        FieldCondition(key="category", match=MatchValue(value="legal")),
        FieldCondition(key="year", range=Range(gte=2023)),
    ]
)
```

## Hybrid Search (Dense + Sparse)

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# BM25 (sparse)
bm25_retriever = BM25Retriever.from_documents(chunks, k=4)

# Dense
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Combine with Reciprocal Rank Fusion
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.4, 0.6],
)
docs = hybrid_retriever.invoke("exact term AND conceptual query")
```

## Multi-Query Retrieval

Generate multiple perspectives of the question to improve recall:

```python
from langchain.retrievers import MultiQueryRetriever
from langchain_openai import ChatOpenAI

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    llm=ChatOpenAI(model="gpt-4o-mini"),
)
docs = multi_retriever.invoke("How does the system handle errors?")
```

## Re-Ranking

Score and re-order initial results for higher precision:

```python
# Cohere re-ranker
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever

reranker = CohereRerank(model="rerank-v3.5", top_n=4)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
)
docs = compression_retriever.invoke("What is the refund policy?")
```

```python
# Cross-encoder re-ranker (local, free)
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
reranker = CrossEncoderReranker(model=model, top_n=4)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
)
```

## HyDE (Hypothetical Document Embeddings)

```python
from langchain.chains import HypotheticalDocumentEmbedder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    base_embeddings=OpenAIEmbeddings(),
    prompt_key="web_search",  # or custom prompt
)
# Use hyde_embeddings as your embedding function in the vector store
```

## Tuning Parameters

| Parameter | Low Value | High Value | Effect |
|-----------|-----------|------------|--------|
| `k` (top-k) | 2–3 | 10–20 | More context vs. noise |
| `score_threshold` | 0.5 | 0.9 | Recall vs. precision |
| `lambda_mult` (MMR) | 0.0 | 1.0 | Diversity vs. relevance |
| `fetch_k` (MMR) | k | 5×k | Candidate pool size |

## Checklist

- [ ] Retrieval strategy matches query patterns (semantic, keyword, or hybrid)
- [ ] Top-k tuned to balance context quality and LLM context window
- [ ] Metadata filters applied when applicable
- [ ] Re-ranking evaluated for precision-critical use cases
- [ ] MMR considered for reducing redundancy
- [ ] Query transformation tested for ambiguous queries
