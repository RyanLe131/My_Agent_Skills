"""
LLM Interface Layer
====================
Unified adapter for communicating with different LLM providers.
Swap providers without changing pipeline code.

Usage:
    from llm import create_llm, create_embeddings

    llm = create_llm("openai")            # or "gemini", "claude"
    embeddings = create_embeddings("openai")

    # Use in RAG pipeline
    response = llm.invoke("What is RAG?")
    vectors = embeddings.embed_documents(["chunk1", "chunk2"])
"""

from llm.base import BaseLLMAdapter
from llm.factory import create_llm, create_embeddings, list_providers

__all__ = [
    "BaseLLMAdapter",
    "create_llm",
    "create_embeddings",
    "list_providers",
]
