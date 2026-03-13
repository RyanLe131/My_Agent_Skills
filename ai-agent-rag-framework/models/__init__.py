"""
Local LLM Models
================
Run LLM models locally without API calls.
Supports Ollama, HuggingFace Transformers, llama.cpp, and vLLM.

Usage:
    from models import create_local_llm, create_local_embeddings, list_backends

    llm = create_local_llm("ollama", model="llama3.1")
    embeddings = create_local_embeddings("huggingface", model="BAAI/bge-large-en-v1.5")
"""

from models.local_factory import (
    create_local_llm,
    create_local_embeddings,
    list_backends,
)

__all__ = [
    "create_local_llm",
    "create_local_embeddings",
    "list_backends",
]
