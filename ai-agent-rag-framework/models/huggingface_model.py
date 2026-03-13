"""
HuggingFace Transformers Backend
=================================
Run open-source models locally via HuggingFace Transformers.
Full control over model loading, quantization, and GPU allocation.

Requires:
    pip install langchain-huggingface transformers torch accelerate
    Optional: pip install bitsandbytes  (for 4-bit/8-bit quantization)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HuggingFaceConfig:
    """Configuration for HuggingFace models."""

    model: str = "meta-llama/Llama-3.1-8B-Instruct"
    device: str = "auto"             # "auto", "cuda", "cpu", "mps"
    max_new_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 0.9
    quantization: str | None = None  # None, "4bit", "8bit"
    trust_remote_code: bool = False
    torch_dtype: str = "auto"        # "auto", "float16", "bfloat16"
    extra: dict = field(default_factory=dict)


def create_llm(config: HuggingFaceConfig | None = None):
    """Create a HuggingFace Transformers LLM instance.

    Args:
        config: Model configuration. Uses defaults if omitted.

    Returns:
        LangChain HuggingFacePipeline instance.

    Example:
        llm = create_llm()
        llm = create_llm(HuggingFaceConfig(
            model="mistralai/Mistral-7B-Instruct-v0.3",
            quantization="4bit",
        ))
    """
    from langchain_huggingface import HuggingFacePipeline
    import torch

    if config is None:
        config = HuggingFaceConfig()

    pipeline_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature if config.temperature > 0 else None,
        "top_p": config.top_p,
        "do_sample": config.temperature > 0,
    }

    model_kwargs = {
        "trust_remote_code": config.trust_remote_code,
    }

    # Set dtype
    if config.torch_dtype == "float16":
        model_kwargs["torch_dtype"] = torch.float16
    elif config.torch_dtype == "bfloat16":
        model_kwargs["torch_dtype"] = torch.bfloat16

    # Quantization
    if config.quantization == "4bit":
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    elif config.quantization == "8bit":
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    # Device mapping
    if config.device == "auto":
        model_kwargs["device_map"] = "auto"
    elif config.device != "cpu":
        model_kwargs["device_map"] = config.device

    return HuggingFacePipeline.from_model_id(
        model_id=config.model,
        task="text-generation",
        pipeline_kwargs=pipeline_kwargs,
        model_kwargs=model_kwargs,
    )


def create_embeddings(
    model: str = "BAAI/bge-large-en-v1.5",
    device: str = "auto",
    normalize: bool = True,
):
    """Create HuggingFace local embeddings.

    Args:
        model: HuggingFace model name or local path.
        device: Device for inference ("auto", "cuda", "cpu", "mps").
        normalize: Normalize embeddings for cosine similarity.

    Returns:
        LangChain HuggingFaceEmbeddings instance.

    Recommended models:
        - BAAI/bge-large-en-v1.5     (1024d, top MTEB)
        - BAAI/bge-small-en-v1.5     (384d, lightweight)
        - sentence-transformers/all-MiniLM-L6-v2  (384d, fast)
        - nomic-ai/nomic-embed-text-v1.5  (768d, long context)
        - Alibaba-NLP/gte-large-en-v1.5   (1024d, strong)
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    return HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": normalize},
    )


# --- VRAM Requirements ---
VRAM_GUIDE = {
    "7B (fp16)": "~14 GB",
    "7B (4-bit)": "~4 GB",
    "7B (8-bit)": "~8 GB",
    "13B (fp16)": "~26 GB",
    "13B (4-bit)": "~8 GB",
    "70B (fp16)": "~140 GB",
    "70B (4-bit)": "~40 GB",
    "Embeddings (any)": "~1-2 GB",
}
