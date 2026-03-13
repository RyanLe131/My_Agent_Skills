"""
Anthropic Claude LLM Adapter
=============================
Adapter for Claude Opus 4, Claude Sonnet 4, Claude Haiku, and Voyage AI embeddings.

Requires:
    pip install langchain-anthropic langchain-voyageai
    Environment variables: ANTHROPIC_API_KEY, VOYAGE_API_KEY
"""

from __future__ import annotations

from llm.base import BaseLLMAdapter, LLMConfig


class ClaudeAdapter(BaseLLMAdapter):
    """Anthropic Claude adapter for LangChain integration."""

    DEFAULT_LLM = "claude-sonnet-4-20250514"
    DEFAULT_EMBEDDINGS = "voyage-3"

    @property
    def provider_name(self) -> str:
        return "claude"

    def get_llm(self):
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=self.config.model or self.DEFAULT_LLM,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            streaming=self.config.streaming,
            **self.config.extra,
        )

    def get_embeddings(self):
        """Claude uses Voyage AI for embeddings (recommended by Anthropic)."""
        from langchain_voyageai import VoyageAIEmbeddings

        model = self.config.extra.get("embedding_model", self.DEFAULT_EMBEDDINGS)
        return VoyageAIEmbeddings(model=model)
