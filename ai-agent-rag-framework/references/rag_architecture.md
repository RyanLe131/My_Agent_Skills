# RAG Architecture

System design patterns for production retrieval augmented generation systems.

## Core Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Data Plane │     │ Serving Plane│     │  Eval Plane  │
│              │     │              │     │              │
│ • Ingestion  │     │ • API / Chat │     │ • Metrics    │
│ • Chunking   │     │ • Retrieval  │     │ • Monitoring │
│ • Embedding  │     │ • Generation │     │ • Feedback   │
│ • Indexing   │     │ • Streaming  │     │ • Iteration  │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       └────────────────────┼────────────────────┘
                            │
                   ┌────────▼────────┐
                   │  Vector Store   │
                   │  + Metadata DB  │
                   └─────────────────┘
```

## Design Patterns

### 1. Naive RAG

The simplest pattern — direct retrieval and generation.

```
Query → Embed → Vector Search (top-k) → Stuff into Prompt → LLM → Answer
```

**Use when**: Prototyping, simple Q&A, small knowledge bases.
**Limitation**: No query refinement, no re-ranking, single retrieval pass.

### 2. Advanced RAG

Adds pre-retrieval and post-retrieval processing.

```
Query → [Query Transform] → Embed → Vector Search → [Re-rank] → Prompt → LLM → Answer
```

Query transformations:
- **Multi-query**: Generate 3-5 query variants, merge results
- **HyDE**: Generate hypothetical answer, embed that instead
- **Step-back**: Abstract the question for broader retrieval

Post-retrieval:
- **Re-ranking**: Cross-encoder scoring of retrieved docs
- **Compression**: Extract only relevant sentences from each chunk
- **Dedup**: Remove near-duplicate chunks

### 3. Modular RAG

Composable pipeline with swappable components.

```
Query → Router → [Search Module A | Search Module B] → Fusion → Re-rank → Generate
```

**Use when**: Multiple data sources, different retrieval strategies per source.

### 4. Agentic RAG

LLM decides when and how to retrieve.

```
Query → Agent → [Decide: retrieve? | answer directly? | clarify?]
                    ↓
              Retrieve → Evaluate sufficiency → [Retrieve more? | Generate]
```

**Use when**: Complex questions requiring multi-step reasoning, tool use.

## Production Considerations

### Latency Budget

| Component | Target (p95) | Optimization |
|-----------|-------------|-------------|
| Embedding | < 100ms | Batch, cache |
| Retrieval | < 200ms | HNSW index, pre-filter |
| Re-ranking | < 300ms | Limit candidates, distilled model |
| Generation | < 2s (first token) | Streaming, smaller model |
| **Total** | **< 3s** | Pipeline parallel where possible |

### Scaling Patterns

- **Read-heavy**: Replica vector DB nodes, cached embeddings
- **Write-heavy**: Async ingestion queue, batch indexing
- **Multi-tenant**: Namespace/collection per tenant, metadata filtering
- **Global**: Regional vector DB deployments, edge caching

### Security

- Validate and sanitize all user queries before embedding
- Use parameterized queries for metadata filtering
- Store API keys and credentials in environment variables
- Implement access control at the retrieval layer for multi-tenant systems
- Rate-limit API endpoints to prevent abuse
- Log queries and answers for audit (with PII handling)
