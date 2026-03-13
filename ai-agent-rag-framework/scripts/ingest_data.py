"""
Document Ingestion Script
=========================
Load documents from various sources into a normalized format with metadata.
Supports: files (PDF, MD, TXT, DOCX, CSV), directories, and web URLs.

Usage:
    from scripts.ingest_data import ingest_directory, ingest_files

    docs = ingest_directory("./my_documents")
    docs = ingest_files(["report.pdf", "guide.md"])
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from datetime import datetime, timezone

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Mapping of file extensions to LangChain loaders
LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".csv": CSVLoader,
}


def ingest_directory(
    path: str,
    glob: str = "**/*",
    extensions: list[str] | None = None,
    metadata: dict | None = None,
) -> list[Document]:
    """Load all supported documents from a directory.

    Args:
        path: Directory path to scan.
        glob: Glob pattern for file matching.
        extensions: Restrict to these file extensions (e.g., [".pdf", ".md"]).
        metadata: Extra metadata to attach to every document.

    Returns:
        List of LangChain Document objects with content and metadata.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")

    allowed_exts = extensions or list(LOADER_MAP.keys())
    all_docs: list[Document] = []

    for ext, loader_cls in LOADER_MAP.items():
        if ext not in allowed_exts:
            continue
        try:
            loader = DirectoryLoader(
                str(path),
                glob=f"**/*{ext}",
                loader_cls=loader_cls,
                show_progress=True,
                use_multithreading=True,
            )
            docs = loader.load()
            logger.info("Loaded %d documents with extension %s", len(docs), ext)
            all_docs.extend(docs)
        except Exception:
            logger.exception("Failed to load %s files from %s", ext, path)

    # Enrich metadata
    for doc in all_docs:
        doc.metadata["ingested_at"] = datetime.now(timezone.utc).isoformat()
        if metadata:
            doc.metadata.update(metadata)

    logger.info("Total documents ingested: %d", len(all_docs))
    return all_docs


def ingest_files(
    file_paths: list[str],
    metadata: dict | None = None,
) -> list[Document]:
    """Load specific files by path.

    Args:
        file_paths: List of file paths to load.
        metadata: Extra metadata to attach to every document.

    Returns:
        List of LangChain Document objects.
    """
    all_docs: list[Document] = []

    for file_path in file_paths:
        p = Path(file_path)
        if not p.is_file():
            logger.warning("File not found, skipping: %s", file_path)
            continue

        ext = p.suffix.lower()
        loader_cls = LOADER_MAP.get(ext)
        if loader_cls is None:
            logger.warning("Unsupported file type %s, skipping: %s", ext, file_path)
            continue

        try:
            docs = loader_cls(str(p)).load()
            for doc in docs:
                doc.metadata["source"] = str(p)
                doc.metadata["ingested_at"] = datetime.now(timezone.utc).isoformat()
                if metadata:
                    doc.metadata.update(metadata)
            all_docs.extend(docs)
        except Exception:
            logger.exception("Failed to load: %s", file_path)

    logger.info("Total documents ingested: %d", len(all_docs))
    return all_docs


def clean_documents(docs: list[Document]) -> list[Document]:
    """Normalize whitespace and remove noise from document content."""
    import re

    for doc in docs:
        text = doc.page_content
        text = re.sub(r"\x00", "", text)          # Remove null bytes
        text = re.sub(r"\n{3,}", "\n\n", text)    # Collapse excess newlines
        text = re.sub(r"[ \t]+", " ", text)        # Normalize whitespace
        doc.page_content = text.strip()

    # Filter out empty documents
    docs = [d for d in docs if len(d.page_content) > 10]
    return docs


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ingest_data.py <directory_path>")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)
    source_dir = sys.argv[1]
    documents = ingest_directory(source_dir)
    documents = clean_documents(documents)
    print(f"Ingested and cleaned {len(documents)} documents from {source_dir}")
