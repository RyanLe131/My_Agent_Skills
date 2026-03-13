"""
Ollama Backend
==============
Run local LLMs via Ollama (https://ollama.com).
Easiest way to run models locally — single binary, no Python dependencies.

Setup:
    1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh
    2. Pull a model: ollama pull llama3.1
    3. Ollama runs as a local server on http://localhost:11434

Requires:
    pip install langchain-ollama
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OllamaConfig:
    """Configuration for Ollama models."""

    model: str = "llama3.1"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    num_ctx: int = 4096          # Context window size
    num_gpu: int = -1            # -1 = auto-detect, 0 = CPU only
    top_p: float = 0.9
    repeat_penalty: float = 1.1


def create_llm(config: OllamaConfig | None = None):
    """Create an Ollama LLM instance.

    Args:
        config: Ollama configuration. Uses defaults if omitted.

    Returns:
        LangChain ChatOllama instance.

    Example:
        llm = create_llm()  # llama3.1 on localhost
        llm = create_llm(OllamaConfig(model="mistral", num_ctx=8192))

        response = llm.invoke("What is RAG?")
    """
    from langchain_ollama import ChatOllama

    if config is None:
        config = OllamaConfig()

    return ChatOllama(
        model=config.model,
        base_url=config.base_url,
        temperature=config.temperature,
        num_ctx=config.num_ctx,
        num_gpu=config.num_gpu,
        top_p=config.top_p,
        repeat_penalty=config.repeat_penalty,
    )


def create_embeddings(model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
    """Create Ollama embeddings.

    Args:
        model: Embedding model name. Pull first: `ollama pull nomic-embed-text`
        base_url: Ollama server URL.

    Returns:
        LangChain OllamaEmbeddings instance.

    Available embedding models:
        - nomic-embed-text    (768d, good general purpose)
        - mxbai-embed-large   (1024d, high quality)
        - all-minilm          (384d, lightweight)
    """
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model=model, base_url=base_url)


# --- Popular Ollama Models ---
RECOMMENDED_MODELS = {
    "general": {
        "llama3.1": "Meta Llama 3.1 8B — strong all-around, fast",
        "llama3.1:70b": "Meta Llama 3.1 70B — high quality, needs 40GB+ VRAM",
        "mistral": "Mistral 7B — fast, good for simple tasks",
        "mixtral": "Mixtral 8x7B MoE — strong quality, moderate resources",
    },
    "code": {
        "codellama": "Code Llama 7B — code generation and completion",
        "deepseek-coder-v2": "DeepSeek Coder V2 — strong coding model",
        "qwen2.5-coder": "Qwen 2.5 Coder — multilingual code",
    },
    "small": {
        "phi3": "Microsoft Phi-3 Mini — 3.8B, runs on CPU",
        "gemma2:2b": "Google Gemma 2 2B — tiny, fast",
        "llama3.2:3b": "Llama 3.2 3B — compact, capable",
    },
    "embeddings": {
        "nomic-embed-text": "768d, general purpose embeddings",
        "mxbai-embed-large": "1024d, high quality embeddings",
        "all-minilm": "384d, lightweight embeddings",
    },
}
