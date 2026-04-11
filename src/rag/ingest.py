"""RAG PDF ingestion utilities.

Provides functions to load agricultural PDFs, chunk them with recursive
character splitting, embed via HuggingFace, and persist to ChromaDB.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.utils import MANUALS_PATH, VECTOR_DB_PATH


def clean_text(text: str) -> str:
    """Clean whitespace and non-ASCII characters from text.

    Args:
        text: Raw text extracted from PDF.

    Returns:
        Cleaned text with normalized whitespace and ASCII-only content.
    """
    # Replace multiple whitespace with single space
    text = re.sub(r"\s+", " ", text)
    # Remove non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]", "", text)
    return text.strip()


def load_pdf_documents(manuals_path: Path | None = None) -> list:
    """Load all PDF documents from the manuals directory.

    Args:
        manuals_path: Path to PDF directory. Defaults to MANUALS_PATH.

    Returns:
        List of loaded LangChain Document objects.

    Raises:
        FileNotFoundError: If the manuals directory does not exist.
        ValueError: If no PDF files are found.
    """
    if manuals_path is None:
        manuals_path = MANUALS_PATH

    if not manuals_path.exists():
        raise FileNotFoundError(
            f"Manuals directory not found: {manuals_path}"
        )

    loader = DirectoryLoader(
        str(manuals_path),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
    )
    documents = loader.load()

    if not documents:
        raise ValueError(f"No PDF files found in {manuals_path}")

    # Clean text content
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    return documents


def split_documents(
    documents: Sequence,
    chunk_size: int = 700,
    chunk_overlap: int = 150,
) -> list:
    """Split documents into chunks using recursive character splitting.

    Args:
        documents: List of LangChain Document objects.
        chunk_size: Target size for each chunk. Defaults to 700.
        chunk_overlap: Characters to overlap between chunks. Defaults to 150.

    Returns:
        List of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(list(documents))


def ingest_knowledge_base(
    manuals_path: Path | None = None,
    vector_db_path: Path | None = None,
) -> Chroma:
    """Ingest PDFs into ChromaDB vector store.

    Loads all PDFs from manuals_path, cleans and chunks the text,
    generates embeddings via all-MiniLM-L6-v2, and persists to disk.

    Args:
        manuals_path: Path to PDF directory. Defaults to MANUALS_PATH.
        vector_db_path: Path for ChromaDB storage. Defaults to VECTOR_DB_PATH.

    Returns:
        Initialized Chroma vector store with embedded documents.

    Raises:
        FileNotFoundError: If manuals directory is missing.
        ValueError: If no PDF files are found.
    """
    if manuals_path is None:
        manuals_path = MANUALS_PATH
    if vector_db_path is None:
        vector_db_path = VECTOR_DB_PATH

    # Ensure vector store directory exists
    vector_db_path.parent.mkdir(parents=True, exist_ok=True)

    # Load and process documents
    documents = load_pdf_documents(manuals_path)
    doc_count = len(set(doc.metadata.get("source", "") for doc in documents))

    chunks = split_documents(documents)
    chunk_count = len(chunks)

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Create and persist vector store
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(vector_db_path),
    )

    # Audit summary
    print(
        f"Indexed {doc_count} documents into {chunk_count} chunks "
        f"at {vector_db_path}"
    )

    return vector_store


def main() -> None:
    """CLI entry point for knowledge base ingestion."""
    ingest_knowledge_base()


if __name__ == "__main__":
    main()
