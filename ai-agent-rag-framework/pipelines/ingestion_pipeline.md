# Ingestion Pipeline

End-to-end workflow for loading documents into the RAG knowledge base.

## Pipeline Steps

```
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌────────────┐
│  Collect   │ →  │   Clean    │ →  │   Chunk    │ →  │   Embed    │ →  │   Store    │
│  Sources   │    │  & Parse   │    │  Documents │    │   Chunks   │    │  Vectors   │
└────────────┘    └────────────┘    └────────────┘    └────────────┘    └────────────┘
```

## Step 1: Collect Sources

Identify and load all document sources:

```python
from scripts.ingest_data import ingest_directory, ingest_files, clean_documents

# Option A: Full directory
docs = ingest_directory(
    "./knowledge_base",
    extensions=[".pdf", ".md", ".txt"],
    metadata={"project": "my-assistant", "version": "1.0"},
)

# Option B: Specific files
docs = ingest_files([
    "./policies/refund_policy.pdf",
    "./guides/user_guide.md",
])
```

## Step 2: Clean & Normalize

```python
docs = clean_documents(docs)
# Removes null bytes, normalizes whitespace, filters empty docs
```

## Step 3: Chunk

```python
from scripts.chunk_documents import chunk_documents, analyze_chunks

chunks = chunk_documents(
    docs,
    strategy="recursive",
    chunk_size=512,
    overlap=50,
)

# Validate chunk quality
stats = analyze_chunks(chunks)
print(stats)
# {'count': 342, 'min': 45, 'max': 520, 'mean': 380, 'short_chunks': 2}
```

## Step 4: Embed & Store

```python
from scripts.generate_embeddings import build_vectorstore

vectorstore = build_vectorstore(
    chunks,
    provider="chromadb",
    embedding_model="text-embedding-3-small",
    collection_name="my-knowledge-base",
    persist_dir="./vector_db",
)
```

## Incremental Updates

For adding new documents without re-indexing everything:

```python
# Load only new/changed files
new_docs = ingest_files(["./new_document.pdf"])
new_docs = clean_documents(new_docs)
new_chunks = chunk_documents(new_docs)

# Add to existing store
vectorstore = load_vectorstore(provider="chromadb", persist_dir="./vector_db")
vectorstore.add_documents(new_chunks)
```

## Automation

Run ingestion on a schedule or trigger:

```bash
# Cron job (daily at 2am)
0 2 * * * cd /app && python scripts/ingest_data.py ./knowledge_base

# Or as part of CI/CD when docs change
# .github/workflows/ingest.yml
```

## Checklist

- [ ] All source documents identified
- [ ] Metadata schema consistent across sources
- [ ] Cleaning removes noise without losing content
- [ ] Chunk sizes validated with `analyze_chunks()`
- [ ] Vector store persisted and loadable
- [ ] Incremental update process defined
