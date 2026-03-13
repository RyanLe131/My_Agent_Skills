"""
Google Gemini LLM Adapter
=========================
Adapter for Gemini 2.5 Pro, Gemini 2.0 Flash, and Google embedding models.

Requires:
    pip install langchain-google-genai
    Environment variable: GOOGLE_API_KEY
"""

from __future__ import annotations

from llm.base import BaseLLMAdapter, LLMConfig


class GeminiAdapter(BaseLLMAdapter):
    """Google Gemini adapter for LangChain integration."""

    DEFAULT_LLM = "gemini-2.0-flash"
    DEFAULT_EMBEDDINGS = "models/text-embedding-004"

    @property
    def provider_name(self) -> str:
        return "gemini"

    def get_llm(self):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=self.config.model or self.DEFAULT_LLM,
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            streaming=self.config.streaming,
            **self.config.extra,
        )

    def get_embeddings(self):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        model = self.config.extra.get("embedding_model", self.DEFAULT_EMBEDDINGS)
        return GoogleGenerativeAIEmbeddings(model=model)
