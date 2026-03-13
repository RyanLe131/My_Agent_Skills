"""
Document Chunking Script
========================
Split documents into retrieval-sized segments while preserving semantic coherence.

Usage:
    from scripts.chunk_documents import chunk_documents

    chunks = chunk_documents(docs, strategy="recursive", chunk_size=512, overlap=50)
"""

from __future__ import annotations

import logging
from langchain_core.documents import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
)

logger = logging.getLogger(__name__)

STRATEGIES = {
    "recursive": RecursiveCharacterTextSplitter,
    "token": TokenTextSplitter,
}


def chunk_documents(
    docs: list[Document],
    strategy: str = "recursive",
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[Document]:
    """Split documents into chunks using the specified strategy.

    Args:
        docs: List of LangChain Document objects.
        strategy: Splitting strategy — "recursive" or "token".
        chunk_size: Target chunk size in characters (recursive) or tokens (token).
        overlap: Overlap between consecutive chunks.

    Returns:
        List of chunked Document objects with preserved metadata.
    """
    splitter_cls = STRATEGIES.get(strategy)
    if splitter_cls is None:
        raise ValueError(f"Unknown strategy: {strategy}. Use: {list(STRATEGIES)}")

    kwargs = {"chunk_size": chunk_size, "chunk_overlap": overlap}
    if strategy == "recursive":
        kwargs["separators"] = ["\n\n", "\n", ". ", " ", ""]

    splitter = splitter_cls(**kwargs)
    chunks = splitter.split_documents(docs)

    # Add chunk index metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    # Filter tiny chunks
    chunks = [c for c in chunks if len(c.page_content.strip()) >= 30]

    logger.info(
        "Chunked %d documents into %d chunks (strategy=%s, size=%d, overlap=%d)",
        len(docs), len(chunks), strategy, chunk_size, overlap,
    )
    return chunks


def chunk_markdown(
    text: str,
    headers: list[tuple[str, str]] | None = None,
) -> list[Document]:
    """Split markdown text by header hierarchy.

    Args:
        text: Raw markdown string.
        headers: List of (header_marker, metadata_key) tuples.

    Returns:
        List of Document objects split by headers.
    """
    if headers is None:
        headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
    return splitter.split_text(text)


def analyze_chunks(chunks: list[Document]) -> dict:
    """Return statistics about chunk sizes for tuning."""
    sizes = [len(c.page_content) for c in chunks]
    stats = {
        "count": len(sizes),
        "min": min(sizes) if sizes else 0,
        "max": max(sizes) if sizes else 0,
        "mean": sum(sizes) / len(sizes) if sizes else 0,
        "short_chunks": sum(1 for s in sizes if s < 50),
    }
    if stats["short_chunks"] > 0:
        logger.warning("%d chunks are under 50 chars — review splitting", stats["short_chunks"])
    return stats
