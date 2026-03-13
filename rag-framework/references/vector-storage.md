# Vector Storage

Persist embeddings with metadata in a vector database for efficient similarity search.

## Database Selection

| Database | Type | Best For | Scaling | Metadata Filtering |
|----------|------|----------|---------|-------------------|
| ChromaDB | Embedded | Prototyping, small datasets | Single node | Basic |
| FAISS | Library | High-performance local search | Single node (GPU) | Manual |
| Pinecone | Managed | Production SaaS, serverless | Auto-scaling | Rich |
| Weaviate | Self-hosted/Cloud | Hybrid search, multi-modal | Horizontal | GraphQL |
| pgvector | Extension | Postgres-native, SQL workflows | Postgres scaling | Full SQL |
| Qdrant | Self-hosted/Cloud | Filtering-heavy workloads | Horizontal | Advanced |
| Milvus | Self-hosted/Cloud | Large-scale, multi-vector | Horizontal | Rich |

### Decision Criteria

- **Prototyping**: ChromaDB (zero config) or FAISS (fast local)
- **Production SaaS**: Pinecone (managed) or Qdrant Cloud
- **Existing Postgres**: pgvector (no new infra)
- **Hybrid search**: Weaviate (built-in BM25 + dense)
- **Heavy filtering**: Qdrant (optimized for metadata filters)
- **Massive scale**: Milvus or Pinecone

## ChromaDB

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# In-memory (prototyping)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    collection_name="my-rag",
)

# Persistent (development)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db",
    collection_name="my-rag",
)

# Load existing
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings(),
    collection_name="my-rag",
)
```

## Pinecone

```python
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# Initialize
pc = Pinecone()  # Uses PINECONE_API_KEY env var
index_name = "my-rag"

# Create index if not exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # Match embedding model
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Store
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    index_name=index_name,
    namespace="v1",
)
```

## FAISS

```python
from langchain_community.vectorstores import FAISS

# Create
vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())

# Save / Load
vectorstore.save_local("./faiss_index")
vectorstore = FAISS.load_local(
    "./faiss_index",
    OpenAIEmbeddings(),
    allow_dangerous_deserialization=True,  # Only for trusted data
)
```

## pgvector

```python
from langchain_postgres.vectorstores import PGVector

# Connection — use environment variables for credentials
import os
connection = os.environ["DATABASE_URL"]  # e.g. postgresql://user:pass@host:5432/db

vectorstore = PGVector.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    connection=connection,
    collection_name="my_rag",
)
```

## Qdrant

```python
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

client = QdrantClient(url="http://localhost:6333")

vectorstore = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    url="http://localhost:6333",
    collection_name="my-rag",
)
```

## Weaviate

```python
from langchain_weaviate import WeaviateVectorStore
import weaviate

client = weaviate.connect_to_local()  # or connect_to_weaviate_cloud()

vectorstore = WeaviateVectorStore.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    client=client,
    index_name="MyRag",
)
```

## LlamaIndex Vector Stores

```python
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

# ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("my-rag")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
```

## Index Types

| Type | Speed | Recall | Memory | Use Case |
|------|-------|--------|--------|----------|
| Flat | Slow | 100% | High | < 10k vectors, exact search |
| HNSW | Fast | ~99% | High | General purpose, most common |
| IVF | Fast | ~95% | Medium | Large datasets, memory constrained |
| PQ | Fastest | ~90% | Low | Massive datasets, approximate OK |

## Checklist

- [ ] Vector store selected based on scale, infrastructure, and query needs
- [ ] Index dimensions match embedding model output
- [ ] Distance metric set correctly (cosine, L2, dot product)
- [ ] Metadata schema matches ingestion metadata fields
- [ ] Database credentials stored in environment variables, not code
- [ ] Persistence configured for non-ephemeral deployments
- [ ] Index type chosen based on dataset size and recall requirements
