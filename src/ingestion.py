"""
Utility script to load test documents with metadata
and split them into overlapping semantic chunks.

Supported document formats:
- .txt, .py, .md
- .docx
- .html, .htm
- .pdf
"""

import json
import hashlib
from pathlib import Path
from langchain_core.documents import Document
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

TEST_DOCUMENTS_DIR = Path(__file__).parent.parent / "test_docs"
METADATA_FILE = TEST_DOCUMENTS_DIR / "document_metadata.json"


def _read_content_by_format(file_path: Path, filename: str) -> str:
    """
    Read content using appropriate loader for the file format
    """
    suffix = file_path.suffix.lower()
    if suffix in (".txt", ".py", ".md"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    if suffix == ".docx":
        try:
            import docx
            doc = docx.Document(str(file_path))
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        except ImportError:
            raise FileNotFoundError(
                f"Cannot load {filename}: python-docx not installed. pip install python-docx")
    
    if suffix in (".html", ".htm"):
        with open(file_path, "r", encoding="utf-8") as f:
            html = f.read()
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style"]):
                tag.decompose()
            return soup.get_text(separator="\n", strip=True)
        except ImportError:
            import re
            return re.sub(r"<[^>]+>", " ", html).strip()
    
    # PDF documents
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(file_path))

            pages = []

            for page in reader.pages:
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(text)

            return "\n".join(pages)

        except ImportError:
            raise ImportError(
                "Cannot load .pdf file: pypdf is not installed. "
                "Install it with: pip install pypdf"
            )

    # Fallback: try text
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def load_documents() -> list:
    """
    Load raw documents from disk.

    Returns:
        List of dictionaries with:
            - "id": str
            - "source_type": str
            - "content": str
            - "metadata": dict
    """

    documents = []

    with open(METADATA_FILE,"r",encoding="utf-8") as f:
        all_metadata = json.load(f)

    for filename, metadata in all_metadata.items():
        file_path = TEST_DOCUMENTS_DIR / filename
        if not file_path.exists():
            print(f"Warning: File not found: {filename}")
            continue
        try:
            content = _read_content_by_format(file_path,filename)
            suffix = file_path.suffix.lower()
            source_type_map = {
                ".pdf": "pdf",
                ".docx": "docx",
                ".html": "html",
                ".htm": "html",
                ".txt": "txt",
                ".py": "python",
                ".md": "markdown",
            }

            source_type = source_type_map.get(suffix,suffix.lstrip(".") or "unknown")

            document = {
                "id": filename,
                "source_type": source_type,
                "content": content,
                "metadata": metadata,
            }
            documents.append(document)

        except (ImportError, OSError, ValueError) as e:
            print(
                f"Warning: Could not load {filename}: {e}"
            )
            continue

    return documents


def chunk_documents(documents: list) -> list:
    """
    Convert full documents into overlapping semantic chunks.

    Uses RecursiveCharacterTextSplitter with:
        - chunk_size = 100 characters
        - chunk_overlap = 20 characters

    Args:
        documents:
            Output from load_documents()

    Returns:
        List of dictionaries with:
            - "doc_id": str
            - "chunk_id": str
            - "text": str
            - "metadata": dict

        Metadata includes:
            - source_type
            - position
            - chunk_size
            - source_file
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n","\n"," ",""])

    chunks = []

    for document in documents:

        doc_id = document["id"]
        source_type = document["source_type"]
        text = document["content"]

        # Skip empty documents
        if not text or not text.strip():
            continue

        # Generate chunks
        split_chunks = text_splitter.split_text(text)

        # Create metadata for each chunk
        for position, chunk_text in enumerate(split_chunks):

            # Create a stable hash for the chunk
            chunk_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:12]

            chunk_id = f"{doc_id}_{position}_{chunk_hash}"

            chunk_metadata = {
                # Preserve original metadata
                **document.get("metadata", {}),

                # Required metadata
                "source_type": source_type,
                "position": position,

                # Access control
                "min_role": document.get(
                    "metadata",
                    {}
                ).get(
                    "min_role",
                    "employee"
                ),

                # Additional metadata
                "source_file": doc_id,
                "chunk_size": len(chunk_text),
                "total_chunks": len(split_chunks),
            }

            chunks.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "metadata": chunk_metadata,
                }
            )

    return chunks


# if __name__ == "__main__":

#     # -------------------------------------------------
#     # Layer 1: Document ingestion
#     # -------------------------------------------------

#     documents = load_documents()

#     print("\n" + "=" * 100)
#     print("DOCUMENT INGESTION")
#     print("=" * 100)

#     print(f"Loaded {len(documents)} documents\n")

#     for document in documents:
#         print(f"ID: {document['id']}")
#         print(f"Source Type: {document['source_type']}")
#         print(f"Content Length: {len(document['content'])}")
#         print("-" * 100)

#     # -------------------------------------------------
#     # Layer 2: Semantic chunking
#     # -------------------------------------------------

#     chunks = chunk_documents(documents)

#     print("\n" + "=" * 100)
#     print("SEMANTIC CHUNKING")
#     print("=" * 100)

#     print(f"Created {len(chunks)} chunks\n")

#     for chunk in chunks:

#         print(f"Doc ID: {chunk['doc_id']}")
#         print(f"Chunk ID: {chunk['chunk_id']}")
#         print(f"Position: {chunk['metadata']['position']}")
#         print(f"Source Type: {chunk['metadata']['source_type']}")
#         print(f"Chunk Size: {len(chunk['text'])}")
#         print("\nText:")
#         print(chunk["text"])
#         print("-" * 100)