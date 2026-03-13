"""
vLLM Backend
=============
High-throughput local inference server with OpenAI-compatible API.
Best for serving models in production or when you need parallel requests.

Setup:
    pip install vllm
    # Start server:
    vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

    The server exposes an OpenAI-compatible API at http://localhost:8000/v1

Requires:
    pip install langchain-openai  (uses OpenAI client to connect to vLLM server)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VLLMConfig:
    """Configuration for vLLM server connection."""

    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    base_url: str = "http://localhost:8000/v1"
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float = 0.9


def create_llm(config: VLLMConfig | None = None):
    """Connect to a running vLLM server.

    The vLLM server exposes an OpenAI-compatible API, so we use
    ChatOpenAI with a custom base_url.

    Args:
        config: vLLM server configuration. Uses defaults if omitted.

    Returns:
        LangChain ChatOpenAI instance pointed at vLLM server.

    Example:
        # First start vLLM server:
        # vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

        llm = create_llm()
        response = llm.invoke("What is RAG?")
    """
    from langchain_openai import ChatOpenAI

    if config is None:
        config = VLLMConfig()

    return ChatOpenAI(
        model=config.model,
        base_url=config.base_url,
        api_key="not-needed",         # vLLM doesn't require an API key
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
    )


# --- vLLM Server Commands ---
# Start server with common configurations:
#
# Basic:
#   vllm serve meta-llama/Llama-3.1-8B-Instruct
#
# With GPU memory limit:
#   vllm serve meta-llama/Llama-3.1-8B-Instruct --gpu-memory-utilization 0.8
#
# Quantized (AWQ):
#   vllm serve TheBloke/Llama-2-7B-Chat-AWQ --quantization awq
#
# Multi-GPU:
#   vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 4
#
# Custom port:
#   vllm serve meta-llama/Llama-3.1-8B-Instruct --port 9000
