---
name: rag-framework
description: 'Build retrieval augmented generation (RAG) systems end-to-end. Use for: document ingestion, text chunking, embedding generation, vector store setup, semantic retrieval, LLM answer generation, and RAG evaluation. Covers LangChain, LlamaIndex, OpenAI, ChromaDB, Pinecone, FAISS, Weaviate, pgvector, Qdrant workflows.'
argument-hint: 'Describe what stage of the RAG pipeline you need help with (e.g., "chunk PDFs for embedding", "set up Pinecone retrieval")'
---

# RAG Framework

Reusable workflows for building retrieval augmented generation systems — from raw documents to evaluated, production-ready pipelines.

## When to Use

- Building a new RAG pipeline from scratch
- Adding document ingestion to an existing LLM application
- Choosing a chunking strategy for your content type
- Generating and storing embeddings
- Setting up vector databases (ChromaDB, Pinecone, FAISS, Weaviate, pgvector, Qdrant)
- Implementing semantic search and context retrieval
- Connecting retrieved context to LLM generation
- Evaluating RAG quality (faithfulness, relevance, recall)

## Pipeline Overview

```
Documents → Ingest → Chunk → Embed → Store → Retrieve → Generate → Evaluate
                                                              ↑          |
                                                              └──────────┘
                                                             (feedback loop)
```

## Procedure

Follow each stage in order for a new pipeline. Jump to any stage for targeted work.

### Stage 1: Ingest Documents

Load raw documents from various sources (files, URLs, databases, APIs) into a normalized format.

**See**: [Ingestion Reference](./references/ingestion.md)

Key decisions:
- Source type (PDF, HTML, Markdown, DOCX, CSV, database)
- Metadata extraction strategy
- Content cleaning and normalization

### Stage 2: Chunk Documents

Split documents into retrieval-sized segments that preserve semantic coherence.

**See**: [Chunking Reference](./references/chunking.md)

Key decisions:
- Chunk size (typically 256–1024 tokens)
- Overlap size (typically 10–20% of chunk size)
- Splitting strategy (recursive, semantic, sentence-based, document-structure)

### Stage 3: Generate Embeddings

Convert text chunks into dense vector representations for similarity search.

**See**: [Embeddings Reference](./references/embeddings.md)

Key decisions:
- Embedding model (OpenAI, Cohere, sentence-transformers, Voyage)
- Dimensionality vs. quality tradeoffs
- Batch processing strategy

### Stage 4: Store Vectors

Persist embeddings in a vector database with metadata for filtered retrieval.

**See**: [Vector Storage Reference](./references/vector-storage.md)

Key decisions:
- Vector database (ChromaDB, Pinecone, FAISS, Weaviate, pgvector, Qdrant)
- Index type (HNSW, IVF, flat)
- Metadata schema design

### Stage 5: Retrieve Context

Query the vector store to find the most relevant chunks for a given user question.

**See**: [Retrieval Reference](./references/retrieval.md)

Key decisions:
- Retrieval method (dense, sparse, hybrid)
- Top-k selection and similarity threshold
- Re-ranking strategy
- Query transformation (HyDE, multi-query, step-back)

### Stage 6: Generate Answers

Feed retrieved context + user query to an LLM to produce grounded answers.

**See**: [Generation Reference](./references/generation.md)

Key decisions:
- Prompt template design
- Context window management
- Citation and source attribution
- Streaming vs. batch responses

### Stage 7: Evaluate Performance

Measure RAG quality across retrieval accuracy, answer faithfulness, and end-to-end relevance.

**See**: [Evaluation Reference](./references/evaluation.md)

Key decisions:
- Metrics (faithfulness, answer relevance, context precision, context recall)
- Evaluation framework (RAGAS, DeepEval, LangSmith, custom)
- Test dataset construction
- Iteration strategy based on results

## Quick Start Patterns

### Minimal RAG (LangChain + ChromaDB + OpenAI)

```python
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 1-2. Ingest & Chunk
loader = DirectoryLoader("./docs", glob="**/*.md")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3-4. Embed & Store
vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())

# 5-6. Retrieve & Generate
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o"),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
)
answer = qa.invoke("What is the main topic?")
```

### Minimal RAG (LlamaIndex + OpenAI)

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 1-4. Ingest, Chunk, Embed, Store
documents = SimpleDirectoryReader("./docs").load_data()
index = VectorStoreIndex.from_documents(documents)

# 5-6. Retrieve & Generate
query_engine = index.as_query_engine()
response = query_engine.query("What is the main topic?")
```
