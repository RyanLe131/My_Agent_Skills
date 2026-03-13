# Document Chunking

Split documents into retrieval-sized segments that preserve meaning and context.

## Strategy Selection

| Strategy | Best For | Pros | Cons |
|----------|----------|------|------|
| Recursive Character | General text | Respects paragraphs/sentences | May split mid-concept |
| Sentence | Conversational text, Q&A | Clean boundaries | Chunks may be too small |
| Semantic | Mixed-topic documents | Concept-aware splits | Requires embeddings, slower |
| Token-based | LLM context management | Precise token counts | Language-agnostic splitting |
| Markdown/HTML | Structured docs | Preserves document structure | Format-specific |
| Code | Source code | Respects functions/classes | Language-specific parsers |
| Parent-Child | Complex retrieval needs | Small chunks + full context | More complex to implement |

## Sizing Guidelines

| Content Type | Chunk Size (tokens) | Overlap | Rationale |
|-------------|-------------------|---------|-----------|
| Technical docs | 512–1024 | 50–100 | Longer context needed |
| Q&A / FAQ | 256–512 | 25–50 | Self-contained answers |
| Legal / contracts | 1024–2048 | 100–200 | Clauses need full context |
| Conversational | 256–512 | 50 | Short, focused turns |
| Code | 512–1024 | 0 | Function/class boundaries |

## LangChain Chunking

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
)

# Recursive (most common)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
)
chunks = splitter.split_documents(docs)

# Token-based (precise for LLM context)
token_splitter = TokenTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    encoding_name="cl100k_base",  # GPT-4 tokenizer
)
chunks = token_splitter.split_documents(docs)

# Markdown-aware (preserves heading hierarchy)
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
)
md_chunks = md_splitter.split_text(markdown_text)
```

## LlamaIndex Chunking

```python
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
    MarkdownNodeParser,
)
from llama_index.embeddings.openai import OpenAIEmbedding

# Sentence-based (default)
parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = parser.get_nodes_from_documents(documents)

# Semantic splitting (groups related sentences)
semantic_parser = SemanticSplitterNodeParser(
    embed_model=OpenAIEmbedding(),
    buffer_size=1,              # Sentences to group for comparison
    breakpoint_percentile_threshold=95,  # Similarity threshold for splits
)
semantic_nodes = semantic_parser.get_nodes_from_documents(documents)

# Markdown-aware
md_parser = MarkdownNodeParser()
md_nodes = md_parser.get_nodes_from_documents(documents)
```

## Parent-Child (Small-to-Big) Pattern

Embed small chunks for precision, retrieve parent chunks for full context:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Small chunks for embedding
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
# Larger chunks returned as context
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), collection_name="parents")
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
retriever.add_documents(docs)
```

## Quality Checks

```python
def analyze_chunks(chunks):
    """Report chunk size distribution for tuning."""
    sizes = [len(c.page_content) for c in chunks]
    print(f"Count: {len(sizes)}")
    print(f"Min: {min(sizes)}, Max: {max(sizes)}, Mean: {sum(sizes)/len(sizes):.0f}")
    # Flag very short chunks (likely noise)
    short = [c for c in chunks if len(c.page_content) < 50]
    if short:
        print(f"Warning: {len(short)} chunks under 50 chars — review splitting")
```

## Checklist

- [ ] Chunking strategy matches content structure
- [ ] Chunk size tuned for embedding model's sweet spot
- [ ] Overlap set to prevent context loss at boundaries
- [ ] Metadata preserved through splitting
- [ ] Chunk size distribution reviewed (no extreme outliers)
- [ ] Small/empty chunks filtered out
