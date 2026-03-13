"""
OpenAI LLM Adapter
==================
Adapter for GPT-4o, GPT-4o-mini, and OpenAI embedding models.

Requires:
    pip install langchain-openai
    Environment variable: OPENAI_API_KEY
"""

from __future__ import annotations

from llm.base import BaseLLMAdapter, LLMConfig


class OpenAIAdapter(BaseLLMAdapter):
    """OpenAI GPT adapter for LangChain integration."""

    # Default models
    DEFAULT_LLM = "gpt-4o"
    DEFAULT_EMBEDDINGS = "text-embedding-3-small"

    @property
    def provider_name(self) -> str:
        return "openai"

    def get_llm(self):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=self.config.model or self.DEFAULT_LLM,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            streaming=self.config.streaming,
            **self.config.extra,
        )

    def get_embeddings(self):
        from langchain_openai import OpenAIEmbeddings

        model = self.config.extra.get("embedding_model", self.DEFAULT_EMBEDDINGS)
        dimensions = self.config.extra.get("embedding_dimensions")
        kwargs = {"model": model}
        if dimensions:
            kwargs["dimensions"] = dimensions
        return OpenAIEmbeddings(**kwargs)
