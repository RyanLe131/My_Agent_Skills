# My_Agent_Skills

A modular collection of reusable AI agent skills for building LLM and RAG-powered applications.

## Skills

### [rag-framework](./rag-framework/)

Lightweight RAG skill with a 7-stage pipeline reference (ingest → chunk → embed → store → retrieve → generate → evaluate). Covers LangChain and LlamaIndex patterns with reference docs for each stage.

### [ai-agent-rag-framework](./ai-agent-rag-framework/)

Production-ready RAG framework with executable code and multi-provider support.

- **`agents/`** — Config files for OpenAI, Gemini, and Claude
- **`llm/`** — Unified LLM adapter layer (swap providers with one argument)
- **`models/`** — Local LLM backends (Ollama, HuggingFace, llama.cpp, vLLM)
- **`scripts/`** — Runnable Python modules for each pipeline stage
- **`templates/`** — System prompts, RAG prompts, answer formats
- **`pipelines/`** — End-to-end workflow guides (ingestion, retrieval, evaluation)
- **`references/`** — Architecture patterns, chunking methods, prompt design, metrics
- **`assets/`** — Example documents and evaluation test sets

See each skill's `SKILL.md` for full documentation.
