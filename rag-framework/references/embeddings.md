# Embedding Generation

Convert text chunks into dense vector representations for similarity search.

## Model Selection

| Model | Dimensions | Context | Provider | Notes |
|-------|-----------|---------|----------|-------|
| `text-embedding-3-small` | 1536 | 8191 | OpenAI | Good cost/quality balance |
| `text-embedding-3-large` | 3072 | 8191 | OpenAI | Best quality, supports dimension reduction |
| `embed-v4.0` | 1024 | 128k | Cohere | Multilingual, long context |
| `all-MiniLM-L6-v2` | 384 | 256 | HuggingFace | Free, fast, lightweight |
| `BAAI/bge-large-en-v1.5` | 1024 | 512 | HuggingFace | Top MTEB, open-source |
| `nomic-embed-text-v1.5` | 768 | 8192 | Nomic | Open-source, long context |
| `voyage-3` | 1024 | 32k | Voyage AI | Strong retrieval performance |
| `mxbai-embed-large-v1` | 1024 | 512 | Mixedbread | Top MTEB leaderboard |

### Decision Criteria

- **Cost-sensitive**: `text-embedding-3-small` or open-source `all-MiniLM-L6-v2`
- **Quality-first**: `text-embedding-3-large` or `BAAI/bge-large-en-v1.5`
- **Multilingual**: Cohere `embed-v4.0`
- **Local/private**: sentence-transformers models via HuggingFace
- **Long documents**: Cohere, Nomic, or Voyage (long context windows)

## OpenAI Embeddings

```python
from langchain_openai import OpenAIEmbeddings

# Standard
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# With dimension reduction (saves storage, minimal quality loss)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    dimensions=1024,  # Reduce from 3072
)

# Direct API usage
from openai import OpenAI

client = OpenAI()
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=["text to embed"],
)
vector = response.data[0].embedding
```

## Local Embeddings (sentence-transformers)

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cuda"},  # or "cpu"
    encode_kwargs={"normalize_embeddings": True},
)
```

## LlamaIndex Embeddings

```python
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# OpenAI
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Local
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5")
```

## Batch Processing

Embed large document sets efficiently:

```python
import time

def batch_embed(texts: list[str], embeddings, batch_size: int = 100):
    """Embed texts in batches with rate limiting."""
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vectors = embeddings.embed_documents(batch)
        all_vectors.extend(vectors)
        if i + batch_size < len(texts):
            time.sleep(0.1)  # Rate limit buffer
    return all_vectors

# Usage
texts = [chunk.page_content for chunk in chunks]
vectors = batch_embed(texts, embeddings)
```

## Caching Embeddings

Avoid re-computing embeddings for unchanged documents:

```python
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore

store = LocalFileStore("./embedding_cache")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=OpenAIEmbeddings(),
    document_embedding_cache=store,
    namespace="my-rag",
)
```

## Checklist

- [ ] Embedding model selected based on quality, cost, and privacy needs
- [ ] Dimensions match vector store configuration
- [ ] Batch size tuned for API rate limits
- [ ] Caching configured for iterative development
- [ ] Same embedding model used for documents and queries
- [ ] Normalization applied if using cosine similarity
