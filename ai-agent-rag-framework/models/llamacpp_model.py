"""
llama.cpp Backend (via llama-cpp-python)
========================================
Run GGUF-quantized models with minimal resources.
Best option for CPU inference and low-VRAM GPUs.

Setup:
    pip install llama-cpp-python
    # With GPU acceleration (CUDA):
    CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python

    Download GGUF models from https://huggingface.co (search for "GGUF")
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LlamaCppConfig:
    """Configuration for llama.cpp models."""

    model_path: str = ""             # Path to .gguf file
    n_ctx: int = 4096                # Context window
    n_gpu_layers: int = -1           # -1 = all layers on GPU, 0 = CPU only
    n_batch: int = 512               # Batch size for prompt processing
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float = 0.9
    verbose: bool = False


def create_llm(config: LlamaCppConfig):
    """Create a llama.cpp LLM instance from a GGUF model file.

    Args:
        config: llama.cpp configuration with model_path set.

    Returns:
        LangChain ChatLlamaCpp instance.

    Example:
        llm = create_llm(LlamaCppConfig(
            model_path="./models/llama-3.1-8b-instruct-q4_k_m.gguf",
            n_gpu_layers=35,  # Offload 35 layers to GPU
        ))
    """
    from langchain_community.chat_models import ChatLlamaCpp

    if not config.model_path:
        raise ValueError("model_path is required — provide path to a .gguf file")

    return ChatLlamaCpp(
        model_path=config.model_path,
        n_ctx=config.n_ctx,
        n_gpu_layers=config.n_gpu_layers,
        n_batch=config.n_batch,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        verbose=config.verbose,
    )


def create_embeddings(model_path: str, n_gpu_layers: int = -1):
    """Create llama.cpp embeddings from a GGUF embedding model.

    Args:
        model_path: Path to a GGUF embedding model file.
        n_gpu_layers: GPU layers to offload (-1 = all).

    Returns:
        LangChain LlamaCppEmbeddings instance.
    """
    from langchain_community.embeddings import LlamaCppEmbeddings

    return LlamaCppEmbeddings(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
    )


# --- Quantization Guide ---
# GGUF quantization levels (quality vs size tradeoff):
#
# | Quant    | Size (7B) | Quality | Speed  | Use Case                    |
# |----------|-----------|---------|--------|-----------------------------|
# | Q2_K     | ~2.7 GB   | Low     | Fast   | Testing only                |
# | Q4_K_M   | ~4.1 GB   | Good    | Fast   | Best balance (recommended)  |
# | Q5_K_M   | ~4.8 GB   | Better  | Medium | Quality-focused             |
# | Q6_K     | ~5.5 GB   | High    | Medium | Near-original quality       |
# | Q8_0     | ~7.2 GB   | Highest | Slow   | When quality matters most   |
# | F16      | ~14 GB    | Perfect | Slow   | Full precision              |
