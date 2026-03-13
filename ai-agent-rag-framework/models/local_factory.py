"""
Local Model Factory
===================
Factory functions to create local LLM and embedding instances by backend name.
Central entry point for all local model inference.

Usage:
    from models.local_factory import create_local_llm, create_local_embeddings

    llm = create_local_llm("ollama", model="llama3.1")
    embeddings = create_local_embeddings("huggingface")
"""

from __future__ import annotations


def create_local_llm(
    backend: str = "ollama",
    model: str | None = None,
    **kwargs,
):
    """Create a local LLM instance.

    Args:
        backend: "ollama", "huggingface", "llamacpp", or "vllm".
        model: Model name or path. Uses backend default if omitted.
        **kwargs: Backend-specific parameters.

    Returns:
        LangChain-compatible LLM instance.

    Examples:
        # Ollama (easiest)
        llm = create_local_llm("ollama", model="llama3.1")

        # HuggingFace with 4-bit quantization
        llm = create_local_llm("huggingface",
            model="meta-llama/Llama-3.1-8B-Instruct",
            quantization="4bit",
        )

        # llama.cpp with GGUF file
        llm = create_local_llm("llamacpp",
            model_path="./models/llama-3.1-8b-q4_k_m.gguf",
            n_gpu_layers=35,
        )

        # vLLM server
        llm = create_local_llm("vllm", model="meta-llama/Llama-3.1-8B-Instruct")
    """
    if backend == "ollama":
        from models.ollama_model import create_llm, OllamaConfig

        config = OllamaConfig(**({"model": model} if model else {}), **kwargs)
        return create_llm(config)

    elif backend == "huggingface":
        from models.huggingface_model import create_llm, HuggingFaceConfig

        config = HuggingFaceConfig(**({"model": model} if model else {}), **kwargs)
        return create_llm(config)

    elif backend == "llamacpp":
        from models.llamacpp_model import create_llm, LlamaCppConfig

        config_kwargs = kwargs
        if model:
            config_kwargs["model_path"] = model
        config = LlamaCppConfig(**config_kwargs)
        return create_llm(config)

    elif backend == "vllm":
        from models.vllm_model import create_llm, VLLMConfig

        config = VLLMConfig(**({"model": model} if model else {}), **kwargs)
        return create_llm(config)

    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. Available: {list_backends()}"
        )


def create_local_embeddings(
    backend: str = "huggingface",
    model: str | None = None,
    **kwargs,
):
    """Create local embeddings instance.

    Args:
        backend: "huggingface" or "ollama" (llamacpp also supported).
        model: Model name or path. Uses backend default if omitted.
        **kwargs: Backend-specific parameters.

    Returns:
        LangChain-compatible Embeddings instance.

    Examples:
        # HuggingFace (best quality)
        embeddings = create_local_embeddings("huggingface", model="BAAI/bge-large-en-v1.5")

        # Ollama (easiest setup)
        embeddings = create_local_embeddings("ollama", model="nomic-embed-text")
    """
    if backend == "huggingface":
        from models.huggingface_model import create_embeddings

        return create_embeddings(model=model or "BAAI/bge-large-en-v1.5", **kwargs)

    elif backend == "ollama":
        from models.ollama_model import create_embeddings

        return create_embeddings(model=model or "nomic-embed-text", **kwargs)

    elif backend == "llamacpp":
        from models.llamacpp_model import create_embeddings

        if not model:
            raise ValueError("model (path to .gguf file) is required for llamacpp embeddings")
        return create_embeddings(model_path=model, **kwargs)

    else:
        raise ValueError(
            f"Unknown embedding backend: {backend!r}. Available: huggingface, ollama, llamacpp"
        )


def list_backends() -> list[str]:
    """Return available local backend names."""
    return ["ollama", "huggingface", "llamacpp", "vllm"]
