# Document Ingestion

Load raw documents from diverse sources into a normalized representation with metadata.

## Loader Selection

| Source | LangChain Loader | LlamaIndex Reader | Notes |
|--------|-----------------|-------------------|-------|
| PDF | `PyPDFLoader`, `UnstructuredPDFLoader` | `PDFReader` | Use Unstructured for complex layouts |
| Markdown | `UnstructuredMarkdownLoader` | `MarkdownReader` | Preserves headers as metadata |
| HTML/Web | `WebBaseLoader`, `RecursiveUrlLoader` | `SimpleWebPageReader` | Respect robots.txt |
| DOCX | `Docx2txtLoader` | `DocxReader` | |
| CSV/Excel | `CSVLoader`, `UnstructuredExcelLoader` | `PandasCSVReader` | Row-per-document or full-table |
| Database | `SQLDatabaseLoader` | `DatabaseReader` | Use parameterized queries to prevent SQL injection |
| API/JSON | `JSONLoader` | `JSONReader` | Use jq-style selectors for nested data |
| Code | `GenericLoader` + `LanguageParser` | `SimpleDirectoryReader` | Language-aware splitting available |
| Images | `UnstructuredImageLoader` | `ImageReader` | OCR via Tesseract or multimodal LLM |

## LangChain Ingestion

```python
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    WebBaseLoader,
    CSVLoader,
)

# Load a directory of PDFs
loader = DirectoryLoader(
    "./documents",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True,
)
docs = loader.load()

# Load from web (always validate/allowlist URLs to prevent SSRF)
ALLOWED_DOMAINS = {"docs.example.com", "wiki.example.com"}
from urllib.parse import urlparse

def safe_load_url(url: str):
    parsed = urlparse(url)
    if parsed.hostname not in ALLOWED_DOMAINS:
        raise ValueError(f"Domain not allowed: {parsed.hostname}")
    return WebBaseLoader(url).load()

# Load CSV with metadata per row
csv_loader = CSVLoader(
    file_path="./data/records.csv",
    metadata_columns=["category", "date"],
)
csv_docs = csv_loader.load()
```

## LlamaIndex Ingestion

```python
from llama_index.core import SimpleDirectoryReader

# Auto-detect file types in a directory
documents = SimpleDirectoryReader(
    input_dir="./documents",
    recursive=True,
    required_exts=[".pdf", ".md", ".txt"],
    filename_as_id=True,
).load_data()

# Add custom metadata
for doc in documents:
    doc.metadata["project"] = "my-rag-system"
```

## Metadata Strategy

Attach metadata during ingestion for downstream filtering:

```python
# Standard metadata fields
metadata = {
    "source": "path/to/file.pdf",      # Origin tracking
    "page": 3,                          # Page number (PDFs)
    "title": "Document Title",          # For citation
    "created_at": "2025-01-15",         # Temporal filtering
    "category": "technical",            # Topic filtering
    "author": "Jane Doe",              # Attribution
}
```

## Content Cleaning

```python
import re

def clean_document(text: str) -> str:
    """Normalize document text before chunking."""
    text = re.sub(r'\n{3,}', '\n\n', text)       # Collapse excess newlines
    text = re.sub(r'[ \t]+', ' ', text)           # Normalize whitespace
    text = re.sub(r'\x00', '', text)              # Remove null bytes
    text = text.strip()
    return text

for doc in docs:
    doc.page_content = clean_document(doc.page_content)
```

## Checklist

- [ ] All source types identified and loaders selected
- [ ] Metadata schema defined (fields, types)
- [ ] Content cleaning applied (whitespace, encoding, null bytes)
- [ ] Large file handling considered (streaming for files > 100MB)
- [ ] External URL loading restricted to allowed domains
- [ ] Database queries use parameterized statements
