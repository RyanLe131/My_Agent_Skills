"""
Base LLM Adapter
================
Abstract interface that all LLM provider adapters must implement.
Ensures consistent behavior across OpenAI, Gemini, Claude, and future providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """Configuration shared across all LLM providers."""

    model: str
    temperature: float = 0.0
    max_tokens: int = 4096
    top_p: float = 1.0
    streaming: bool = False
    extra: dict = field(default_factory=dict)


class BaseLLMAdapter(ABC):
    """Abstract base class for LLM provider adapters.

    Every adapter must provide:
    - get_llm()        → LangChain-compatible LLM instance
    - get_embeddings() → LangChain-compatible Embeddings instance
    - provider_name    → string identifying the provider
    """

    def __init__(self, config: LLMConfig):
        self.config = config

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier (e.g., 'openai', 'gemini', 'claude')."""
        ...

    @abstractmethod
    def get_llm(self):
        """Return a LangChain-compatible LLM instance."""
        ...

    @abstractmethod
    def get_embeddings(self):
        """Return a LangChain-compatible Embeddings instance."""
        ...

    def get_llm_with_fallback(self, fallback_adapter: BaseLLMAdapter):
        """Return an LLM with automatic fallback to another provider on failure."""
        from langchain_core.runnables import RunnableWithFallbacks

        return self.get_llm().with_fallbacks([fallback_adapter.get_llm()])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.config.model!r})"
