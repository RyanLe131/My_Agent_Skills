---
name: ai-agent-rag-framework
description: 'Production-ready RAG framework for AI agents. Use for: document QA, enterprise knowledge assistants, research assistants, internal copilots, support bots. Covers ingestion, chunking, embeddings, vector storage, retrieval, LLM generation, and evaluation pipelines. Supports OpenAI, LangChain, LlamaIndex, ChromaDB, Pinecone, FAISS, pgvector, Qdrant.'
argument-hint: 'Describe what you need (e.g., "build a document QA bot", "set up retrieval pipeline with Pinecone")'
---

# AI Agent RAG Framework

A modular, production-ready skill for building Retrieval Augmented Generation systems across diverse AI agent use cases.

## Target Use Cases

| Use Case | Description | Key Requirements |
|----------|-------------|-----------------|
| Document QA | Answer questions from uploaded documents | Fast ingestion, precise retrieval |
| Enterprise Knowledge Assistant | Query internal wikis, policies, SOPs | Access control, hybrid search |
| Research Assistant | Synthesize answers from papers/articles | Long-context chunking, citation |
| Internal Copilot | Developer/employee productivity tool | Code-aware, multi-source |
| Support Bot | Customer-facing Q&A from knowledge base | Low latency, guardrails, streaming |

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                             │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │  Ingest  │→ │  Chunk   │→ │  Embed   │→ │  Vector Store │  │
│  │  Data    │  │  Docs    │  │  Chunks  │  │  (Index)      │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────┬───────┘  │
│                                                     │          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐          │          │
│  │ Evaluate │← │ Generate │← │ Retrieve │←─────────┘          │
│  │ Quality  │  │ Answer   │  │ Context  │                      │
│  └──────────┘  └──────────┘  └──────────┘                      │
│       │                                                         │
│       └─── feedback loop ──→ tune chunking/retrieval/prompts   │
└─────────────────────────────────────────────────────────────────┘
```

## Standard Workflow

### Stage 1 — Ingest Data

Load raw documents from files, URLs, databases, or APIs into a normalized format with metadata.

- **Script**: [ingest_data.py](./scripts/ingest_data.py)
- **Pipeline**: [Ingestion Pipeline](./pipelines/ingestion_pipeline.md)

### Stage 2 — Chunk Documents

Split documents into retrieval-sized segments preserving semantic coherence.

- **Script**: [chunk_documents.py](./scripts/chunk_documents.py)
- **Reference**: [Chunking Methods](./references/chunking_methods.md)

### Stage 3 — Generate Embeddings

Convert text chunks into dense vectors using an embedding model.

- **Script**: [generate_embeddings.py](./scripts/generate_embeddings.py)

### Stage 4 — Store in Vector Database

Persist embeddings with metadata for efficient similarity search.

- **Reference**: [RAG Architecture](./references/rag_architecture.md)

### Stage 5 — Retrieve Context

Query the vector store to find the most relevant chunks for a user question.

- **Script**: [retrieve_context.py](./scripts/retrieve_context.py)
- **Pipeline**: [Retrieval Pipeline](./pipelines/retrieval_pipeline.md)

### Stage 6 — Generate Answer

Feed retrieved context + user query to an LLM with a structured prompt.

- **Templates**: [System Prompt](./templates/system_prompt.txt) · [RAG Prompt](./templates/rag_prompt.txt) · [Answer Format](./templates/answer_format.md)
- **Reference**: [Prompt Design](./references/prompt_design.md)

### Stage 7 — Evaluate Performance

Measure retrieval and generation quality, then iterate.

- **Script**: [evaluate_answers.py](./scripts/evaluate_answers.py)
- **Pipeline**: [Evaluation Pipeline](./pipelines/evaluation_pipeline.md)
- **Reference**: [Evaluation Metrics](./references/evaluation_metrics.md)
- **Assets**: [Evaluation Questions](./assets/evaluation_questions.json)

## Agent Configuration

| Agent | Model | Embedding Model | Config |
|-------|-------|-----------------|--------|
| **OpenAI** | GPT-4o | text-embedding-3-small | [openai.yaml](./agents/openai.yaml) |
| **Gemini** | Gemini 2.0 Flash | text-embedding-004 | [gemini.yaml](./agents/gemini.yaml) |
| **Claude** | Claude Sonnet 4 | Voyage 3 | [claude.yaml](./agents/claude.yaml) |

Each config includes: model settings, tool definitions, guardrails, monitoring, and LangChain/LlamaIndex integration examples.

## LLM Interface Layer

The `llm/` folder provides a unified adapter layer for communicating with any LLM provider. Pipeline code uses the factory — never imports provider SDKs directly.

```python
from llm import create_llm, create_embeddings

# Swap providers with one argument — no pipeline code changes
llm = create_llm("openai")                          # GPT-4o
llm = create_llm("gemini", model="gemini-2.5-pro")   # Gemini
llm = create_llm("claude", streaming=True)            # Claude

# Embeddings
embeddings = create_embeddings("openai")              # text-embedding-3-small
embeddings = create_embeddings("claude")              # Voyage 3

# Auto-fallback: if primary fails, switch to backup
from llm import create_llm_with_fallback
llm = create_llm_with_fallback(primary="openai", fallback="claude")
```

| File | Purpose |
|------|---------|
| [base.py](./llm/base.py) | Abstract interface (`BaseLLMAdapter`) |
| [factory.py](./llm/factory.py) | `create_llm()`, `create_embeddings()` factory |
| [openai_llm.py](./llm/openai_llm.py) | GPT-4o / text-embedding-3 adapter |
| [gemini_llm.py](./llm/gemini_llm.py) | Gemini 2.0 Flash / text-embedding-004 adapter |
| [claude_llm.py](./llm/claude_llm.py) | Claude Sonnet 4 / Voyage 3 adapter |

## Local LLM Models

The `models/` folder runs LLMs locally — no API keys, full data privacy.

```python
from models import create_local_llm, create_local_embeddings

# Ollama (easiest — one binary, pull models like Docker)
llm = create_local_llm("ollama", model="llama3.1")

# HuggingFace Transformers (full control, quantization)
llm = create_local_llm("huggingface",
    model="meta-llama/Llama-3.1-8B-Instruct",
    quantization="4bit",
)

# llama.cpp (GGUF files, best for CPU / low VRAM)
llm = create_local_llm("llamacpp",
    model_path="./models/llama-3.1-8b-q4_k_m.gguf",
    n_gpu_layers=35,
)

# vLLM (high-throughput server, OpenAI-compatible API)
llm = create_local_llm("vllm", model="meta-llama/Llama-3.1-8B-Instruct")

# Local embeddings (no API needed)
embeddings = create_local_embeddings("huggingface", model="BAAI/bge-large-en-v1.5")
embeddings = create_local_embeddings("ollama", model="nomic-embed-text")
```

| File | Backend | Best For |
|------|---------|----------|
| [ollama_model.py](./models/ollama_model.py) | Ollama | Easiest setup, quick prototyping |
| [huggingface_model.py](./models/huggingface_model.py) | Transformers | Full control, 4/8-bit quantization |
| [llamacpp_model.py](./models/llamacpp_model.py) | llama.cpp | CPU inference, GGUF models |
| [vllm_model.py](./models/vllm_model.py) | vLLM | Production serving, high throughput |
| [local_factory.py](./models/local_factory.py) | Factory | `create_local_llm()` unified entry point |

## Quick Start

```python
from llm import create_llm, create_embeddings
from scripts.ingest_data import ingest_directory
from scripts.chunk_documents import chunk_documents
from scripts.generate_embeddings import build_vectorstore
from scripts.retrieve_context import retrieve
from scripts.evaluate_answers import evaluate_rag

# Choose your LLM provider
llm = create_llm("openai")  # or "gemini", "claude"

# 1-4. Ingest → Chunk → Embed → Store
docs = ingest_directory("./my_documents")
chunks = chunk_documents(docs)
vectorstore = build_vectorstore(chunks)

# 5-6. Retrieve → Generate
context_docs = retrieve(vectorstore, "What is our refund policy?")

# 7. Evaluate
scores = evaluate_rag(test_questions="./assets/evaluation_questions.json")
```

## Folder Structure

```
ai-agent-rag-framework/
├── SKILL.md                           # This file — entry point
├── agents/
│   ├── openai.yaml                    # GPT-4o config (model, tools, guardrails)
│   ├── gemini.yaml                    # Gemini 2.0 Flash config
│   └── claude.yaml                    # Claude Sonnet 4 config
├── llm/                               # LLM interface / adapter layer
│   ├── __init__.py                    # Package exports (create_llm, create_embeddings)
│   ├── base.py                        # Abstract BaseLLMAdapter interface
│   ├── factory.py                     # Provider factory + fallback support
│   ├── openai_llm.py                  # OpenAI GPT adapter
│   ├── gemini_llm.py                  # Google Gemini adapter
│   └── claude_llm.py                  # Anthropic Claude adapter
├── models/                            # Local LLM inference backends
│   ├── __init__.py                    # Package exports (create_local_llm)
│   ├── local_factory.py               # Backend factory (ollama/hf/llamacpp/vllm)
│   ├── ollama_model.py                # Ollama backend (easiest local setup)
│   ├── huggingface_model.py           # HuggingFace Transformers + quantization
│   ├── llamacpp_model.py              # llama.cpp / GGUF backend (CPU-friendly)
│   └── vllm_model.py                  # vLLM server (high-throughput production)
├── scripts/
│   ├── ingest_data.py                 # Stage 1: document loading
│   ├── chunk_documents.py             # Stage 2: text splitting
│   ├── generate_embeddings.py         # Stage 3-4: embed & store
│   ├── retrieve_context.py            # Stage 5: vector search
│   └── evaluate_answers.py            # Stage 7: quality metrics
├── references/
│   ├── rag_architecture.md            # System design patterns
│   ├── chunking_methods.md            # Splitting strategies
│   ├── prompt_design.md               # Prompt engineering for RAG
│   └── evaluation_metrics.md          # Quality measurement
├── templates/
│   ├── system_prompt.txt              # LLM system instruction
│   ├── rag_prompt.txt                 # Context + question template
│   └── answer_format.md              # Output formatting rules
├── pipelines/
│   ├── ingestion_pipeline.md          # End-to-end ingestion flow
│   ├── retrieval_pipeline.md          # Search & re-rank flow
│   └── evaluation_pipeline.md         # Testing & monitoring flow
└── assets/
    ├── example_documents.json         # Sample documents for testing
    └── evaluation_questions.json      # Test Q&A pairs
```
