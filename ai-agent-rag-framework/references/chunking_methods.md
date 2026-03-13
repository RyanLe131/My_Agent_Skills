# Chunking Methods

Strategies for splitting documents into retrieval-optimized segments.

## Strategy Comparison

| Method | How It Works | Best For | Chunk Quality |
|--------|-------------|----------|---------------|
| **Recursive Character** | Split by separators (¶ → \n → . → space) | General text | Good |
| **Sentence** | Split at sentence boundaries | Conversational, Q&A | Good |
| **Semantic** | Group sentences by embedding similarity | Mixed-topic docs | Best |
| **Token** | Split at exact token counts | LLM context control | Medium |
| **Markdown/HTML** | Split at structural elements (headers) | Structured docs | Good |
| **Code** | Split at language constructs (functions, classes) | Source code | Good |
| **Parent-Child** | Small chunks for search, large for context | Complex retrieval | Best |

## Sizing Guide

The right chunk size depends on your content and retrieval needs:

| Factor | Smaller Chunks (128-256) | Larger Chunks (512-1024) |
|--------|-------------------------|-------------------------|
| Retrieval precision | Higher (focused) | Lower (diluted) |
| Context completeness | Lower (fragmented) | Higher (self-contained) |
| Embedding quality | Better (focused signal) | Worse (mixed signals) |
| Number of chunks | More (storage/cost) | Fewer (efficient) |
| Best for | Factual Q&A, lookups | Complex reasoning, summaries |

**Rule of thumb**: Start with 512 tokens, 50 token overlap. Tune based on evaluation.

## Overlap Strategy

Overlap prevents information loss at chunk boundaries:

```
Chunk 1: [============================]
Chunk 2:                     [============================]
                             ^^^^^^^^^^^
                              overlap
```

- **10-20% overlap** is standard (e.g., 50-100 tokens for 512-token chunks)
- **No overlap** for structured content with clear boundaries (code, tables)
- **Higher overlap** for dense, flowing prose (legal, academic)

## Implementation Examples

### Recursive Character (Default)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)
```

Tries `\n\n` first (paragraphs), falls back to `\n` (lines), then sentences, then words.

### Semantic Chunking

Groups consecutive sentences whose embeddings are similar:

```python
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding

parser = SemanticSplitterNodeParser(
    embed_model=OpenAIEmbedding(),
    buffer_size=1,
    breakpoint_percentile_threshold=95,
)
```

Higher `breakpoint_percentile_threshold` = fewer splits (larger chunks).

### Parent-Child (Small-to-Big)

Embed small chunks for precision; return parent chunk for full context:

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore

child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=InMemoryStore(),
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
```

## Anti-Patterns

- **Chunk too small**: Fragments context, retrieves noise
- **Chunk too large**: Dilutes embedding signal, wastes context window
- **Zero overlap**: Loses information at boundaries
- **Ignoring structure**: Splitting tables, code blocks, or lists mid-element
- **One-size-fits-all**: Using the same strategy for PDFs, code, and chat logs
