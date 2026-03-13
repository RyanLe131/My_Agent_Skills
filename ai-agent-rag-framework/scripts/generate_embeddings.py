"""
Embedding Generation & Vector Storage Script
=============================================
Convert text chunks into dense vectors and store in a vector database.

Usage:
    from scripts.generate_embeddings import build_vectorstore, load_vectorstore

    vectorstore = build_vectorstore(chunks, provider="chromadb")
    vectorstore = load_vectorstore(provider="chromadb", persist_dir="./chroma_db")
"""

from __future__ import annotations

import os
import time
import logging
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def get_embeddings(model: str = "text-embedding-3-small", **kwargs):
    """Create an embedding model instance.

    Args:
        model: Embedding model name.
        **kwargs: Additional model parameters (e.g., dimensions).

    Returns:
        LangChain Embeddings instance.
    """
    if model.startswith("text-embedding"):
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model, **kwargs)

    # Fallback to HuggingFace for local models
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=model,
        encode_kwargs={"normalize_embeddings": True},
        **kwargs,
    )


def build_vectorstore(
    chunks: list[Document],
    provider: str = "chromadb",
    embedding_model: str = "text-embedding-3-small",
    collection_name: str = "rag-knowledge-base",
    persist_dir: str = "./vector_db",
    **kwargs,
):
    """Embed chunks and store them in a vector database.

    Args:
        chunks: Chunked documents to embed and store.
        provider: Vector DB provider — "chromadb", "faiss", "pinecone", "pgvector", "qdrant".
        embedding_model: Name of the embedding model.
        collection_name: Name of the vector collection/index.
        persist_dir: Directory for local persistence (chromadb, faiss).
        **kwargs: Additional provider-specific parameters.

    Returns:
        LangChain VectorStore instance.
    """
    embeddings = get_embeddings(embedding_model)

    if provider == "chromadb":
        from langchain_community.vectorstores import Chroma
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name=collection_name,
        )

    elif provider == "faiss":
        from langchain_community.vectorstores import FAISS
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(persist_dir)

    elif provider == "pinecone":
        from langchain_pinecone import PineconeVectorStore
        vectorstore = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=kwargs.get("index_name", collection_name),
            namespace=kwargs.get("namespace", "default"),
        )

    elif provider == "qdrant":
        from langchain_qdrant import QdrantVectorStore
        vectorstore = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            url=kwargs.get("url", "http://localhost:6333"),
            collection_name=collection_name,
        )

    elif provider == "pgvector":
        from langchain_postgres.vectorstores import PGVector
        connection = os.environ["DATABASE_URL"]
        vectorstore = PGVector.from_documents(
            documents=chunks,
            embedding=embeddings,
            connection=connection,
            collection_name=collection_name,
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")

    logger.info(
        "Stored %d chunks in %s (collection=%s)", len(chunks), provider, collection_name
    )
    return vectorstore


def load_vectorstore(
    provider: str = "chromadb",
    embedding_model: str = "text-embedding-3-small",
    collection_name: str = "rag-knowledge-base",
    persist_dir: str = "./vector_db",
    **kwargs,
):
    """Load an existing vector store.

    Args:
        provider: Vector DB provider.
        embedding_model: Must match the model used during build.
        collection_name: Collection/index name.
        persist_dir: Local persistence directory.

    Returns:
        LangChain VectorStore instance.
    """
    embeddings = get_embeddings(embedding_model)

    if provider == "chromadb":
        from langchain_community.vectorstores import Chroma
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=collection_name,
        )

    elif provider == "faiss":
        from langchain_community.vectorstores import FAISS
        return FAISS.load_local(
            persist_dir, embeddings, allow_dangerous_deserialization=True
        )

    else:
        raise ValueError(f"Load not implemented for: {provider}. Use build_vectorstore.")


def batch_embed(texts: list[str], embeddings, batch_size: int = 100) -> list[list[float]]:
    """Embed texts in batches with rate-limit buffering."""
    all_vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vectors = embeddings.embed_documents(batch)
        all_vectors.extend(vectors)
        if i + batch_size < len(texts):
            time.sleep(0.1)
    return all_vectors
