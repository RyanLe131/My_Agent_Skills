"""
LLM Factory
============
Factory functions to create LLM and embedding instances by provider name.
Central entry point — pipeline code uses this instead of importing provider modules directly.

Usage:
    from llm.factory import create_llm, create_embeddings

    llm = create_llm("openai", model="gpt-4o", temperature=0)
    embeddings = create_embeddings("gemini")
"""

from __future__ import annotations

from llm.base import BaseLLMAdapter, LLMConfig

# Registry of provider adapters (lazy imports to avoid requiring all SDKs)
_PROVIDERS: dict[str, str] = {
    "openai": "llm.openai_llm.OpenAIAdapter",
    "gemini": "llm.gemini_llm.GeminiAdapter",
    "claude": "llm.claude_llm.ClaudeAdapter",
}


def _import_adapter(dotted_path: str) -> type[BaseLLMAdapter]:
    """Dynamically import an adapter class from its dotted path."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _get_adapter(
    provider: str,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    streaming: bool = False,
    **kwargs,
) -> BaseLLMAdapter:
    """Instantiate an adapter for the given provider."""
    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider: {provider!r}. Available: {list(_PROVIDERS)}"
        )

    adapter_cls = _import_adapter(_PROVIDERS[provider])
    config = LLMConfig(
        model=model or "",
        temperature=temperature,
        max_tokens=max_tokens,
        streaming=streaming,
        extra=kwargs,
    )
    return adapter_cls(config)


def create_llm(
    provider: str = "openai",
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    streaming: bool = False,
    **kwargs,
):
    """Create a LangChain-compatible LLM instance.

    Args:
        provider: "openai", "gemini", or "claude".
        model: Model name (uses provider default if omitted).
        temperature: Sampling temperature (0 = deterministic).
        max_tokens: Maximum output tokens.
        streaming: Enable token streaming.
        **kwargs: Provider-specific parameters.

    Returns:
        LangChain ChatModel instance.

    Example:
        llm = create_llm("openai", model="gpt-4o-mini")
        llm = create_llm("claude", streaming=True)
        llm = create_llm("gemini", model="gemini-2.5-pro")
    """
    adapter = _get_adapter(provider, model, temperature, max_tokens, streaming, **kwargs)
    return adapter.get_llm()


def create_embeddings(
    provider: str = "openai",
    embedding_model: str | None = None,
    **kwargs,
):
    """Create a LangChain-compatible Embeddings instance.

    Args:
        provider: "openai", "gemini", or "claude".
        embedding_model: Override the default embedding model.
        **kwargs: Provider-specific parameters.

    Returns:
        LangChain Embeddings instance.

    Example:
        embeddings = create_embeddings("openai")
        embeddings = create_embeddings("openai", embedding_model="text-embedding-3-large", embedding_dimensions=1024)
        embeddings = create_embeddings("gemini")
        embeddings = create_embeddings("claude")  # Uses Voyage AI
    """
    if embedding_model:
        kwargs["embedding_model"] = embedding_model
    adapter = _get_adapter(provider, **kwargs)
    return adapter.get_embeddings()


def create_llm_with_fallback(
    primary: str = "openai",
    fallback: str = "claude",
    **kwargs,
):
    """Create an LLM that automatically falls back to another provider on error.

    Args:
        primary: Primary provider name.
        fallback: Fallback provider name.
        **kwargs: Shared parameters for both providers.

    Returns:
        LangChain Runnable with fallback.
    """
    primary_adapter = _get_adapter(primary, **kwargs)
    fallback_adapter = _get_adapter(fallback, **kwargs)
    return primary_adapter.get_llm_with_fallback(fallback_adapter)


def list_providers() -> list[str]:
    """Return available provider names."""
    return list(_PROVIDERS.keys())
