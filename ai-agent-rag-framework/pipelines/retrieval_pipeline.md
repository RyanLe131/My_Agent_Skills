# Retrieval Pipeline

End-to-end workflow for finding relevant context and generating answers.

## Pipeline Steps

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  User    │ →  │  Query   │ →  │  Vector  │ →  │ Re-rank  │ →  │ Generate │
│  Query   │    │ Transform│    │  Search  │    │ & Filter │    │  Answer  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

## Step 1: Receive Query

```python
user_query = "What is our company's vacation policy?"
```

## Step 2: Query Transformation (Optional)

Improve retrieval by rephrasing or expanding the query:

```python
# Multi-query: generate variations
from langchain.retrievers import MultiQueryRetriever
multi_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(), llm=llm
)

# HyDE: embed a hypothetical answer instead
from langchain.chains import HypotheticalDocumentEmbedder
hyde = HypotheticalDocumentEmbedder.from_llm(llm=llm, base_embeddings=embeddings)
```

## Step 3: Vector Search

```python
from scripts.retrieve_context import retrieve

docs = retrieve(
    vectorstore,
    query=user_query,
    search_type="mmr",       # similarity | mmr | similarity_score_threshold
    top_k=5,
    filters={"category": "hr_policies"},  # optional metadata filter
)
```

## Step 4: Re-rank & Filter (Optional)

```python
from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever

reranker = CohereRerank(model="rerank-v3.5", top_n=3)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=vectorstore.as_retriever(search_kwargs={"k": 20}),
)
docs = compression_retriever.invoke(user_query)
```

## Step 5: Generate Answer

```python
from scripts.retrieve_context import create_rag_chain

chain = create_rag_chain(
    vectorstore,
    model="gpt-4o",
    top_k=5,
)
answer = chain.invoke(user_query)
print(answer)
```

## Full Pipeline (One-Shot)

```python
from scripts.generate_embeddings import load_vectorstore
from scripts.retrieve_context import create_rag_chain

# Load existing knowledge base
vectorstore = load_vectorstore(provider="chromadb", persist_dir="./vector_db")

# Create and run chain
chain = create_rag_chain(vectorstore)
answer = chain.invoke("What is the vacation policy?")
```

## Tuning Guide

| Symptom | Cause | Fix |
|---------|-------|-----|
| Irrelevant results | Poor embedding match | Try hybrid search, re-ranking |
| Missing results | Top-k too low | Increase top-k, add multi-query |
| Redundant results | No diversity | Use MMR search type |
| Slow responses | Too many candidates | Reduce fetch_k, pre-filter by metadata |
