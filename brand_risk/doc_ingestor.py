"""Ingest supplier contract documents (PDF or DOCX) into the contracts RAG collection.

Writes uploaded bytes to a NamedTemporaryFile so LangChain's file-path-based
loaders can process them, then cleans up. Documents are chunked at clause-level
granularity before indexing so individual clause retrieval works reliably.

Public API:
    ingest(file_bytes, filename, vendor_id) -> int   # returns chunk count indexed
"""
from __future__ import annotations

import logging
import os
import tempfile

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .rag import index_contract

logger = logging.getLogger(__name__)

CHUNK_SIZE    = 800   # characters — sized for clause-level granularity
CHUNK_OVERLAP = 120   # overlap prevents mid-clause cuts

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " "],
)


def ingest(file_bytes: bytes, filename: str, vendor_id: str) -> int:
    """Load, chunk, and index a supplier contract into the contracts RAG collection.

    Args:
        file_bytes: Raw bytes of the uploaded PDF or DOCX file.
        filename:   Original filename — determines loader and is stored as source metadata.
        vendor_id:  Entity ID of the vendor this contract belongs to.

    Returns:
        Number of chunks indexed into the Chroma contracts collection.

    Raises:
        ValueError: If the file extension is not pdf, docx, or doc.
    """
    ext = filename.rsplit(".", 1)[-1].lower()
    raw_docs = _load(file_bytes, filename, ext)
    chunks = _chunk(raw_docs, vendor_id, filename)
    index_contract(chunks, vendor_id)
    logger.info("Ingested '%s' → %d chunks for vendor '%s'", filename, len(chunks), vendor_id)
    return len(chunks)


def _load(file_bytes: bytes, filename: str, ext: str) -> list[Document]:
    """Write bytes to a temp file, load via LangChain loader, then delete temp file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        if ext == "pdf":
            loader = PyPDFLoader(tmp_path)
        elif ext in ("docx", "doc"):
            loader = Docx2txtLoader(tmp_path)
        else:
            raise ValueError(f"Unsupported format '.{ext}'. Upload a PDF or DOCX file.")
        return loader.load()
    finally:
        os.unlink(tmp_path)


def ingest_playbook(file_bytes: bytes, filename: str) -> int:
    """Load, chunk, and index a crisis response playbook PDF/DOCX.

    Args:
        file_bytes: Raw bytes of the uploaded PDF or DOCX file.
        filename:   Original filename — determines loader and is stored as source metadata.

    Returns:
        Number of chunks indexed into the Chroma playbook collection.
    """
    from .rag_playbook import index_playbook
    ext = filename.rsplit(".", 1)[-1].lower()
    raw_docs = _load(file_bytes, filename, ext)
    chunks = _chunk(raw_docs, vendor_id="__playbook__", source=filename)
    index_playbook(chunks)
    logger.info("Ingested playbook '%s' → %d chunks", filename, len(chunks))
    return len(chunks)


def _chunk(docs: list[Document], vendor_id: str, source: str) -> list[Document]:
    """Split documents into clause-sized chunks and stamp vendor_id onto metadata."""
    chunks = _SPLITTER.split_documents(docs)
    for chunk in chunks:
        chunk.metadata["vendor_id"] = vendor_id
        chunk.metadata.setdefault("source", source)
    return chunks
