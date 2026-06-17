"""Contracts Chroma collection — extracted from rag.py to keep modules under 200 lines.

Holds one Chroma collection ("contracts") shared across all vendor uploads.
Metadata filter {"vendor_id": vendor_id} at retrieval time isolates clauses to one vendor.

Public API:
    retrieve_contract_clauses(vendor_id, context, k) -> list[str]
    index_contract(documents, vendor_id)             -> None
"""
from __future__ import annotations

import logging

from langchain_core.documents import Document

from .rag import CHROMA_DIR, _get_embeddings

logger = logging.getLogger(__name__)

CONTRACTS_COLLECTION: str = "contracts"

_contract_store = None  # lazy singleton


def _get_contract_store():
    """Lazy-load the contracts Chroma collection. Returns None if absent or empty."""
    global _contract_store
    if _contract_store is not None:
        return _contract_store
    import os
    if not os.path.exists(CHROMA_DIR):
        return None
    try:
        from langchain_chroma import Chroma
        store = Chroma(
            collection_name=CONTRACTS_COLLECTION,
            embedding_function=_get_embeddings(),
            persist_directory=CHROMA_DIR,
        )
        if store._collection.count() == 0:
            return None
        _contract_store = store
        logger.info("Contracts store loaded: %d chunks", store._collection.count())
        return _contract_store
    except Exception as exc:
        logger.warning("Could not load contracts store (%s); clause RAG disabled", exc)
        return None


def retrieve_contract_clauses(vendor_id: str, context: str, k: int = 3) -> list[str]:
    """Return k relevant contract clause excerpts for a given vendor.

    Uses a Chroma metadata filter {"vendor_id": vendor_id} so only chunks from
    that vendor's contract are returned, not another supplier's document.
    Returns [] when no contract has been indexed for this vendor.

    Args:
        vendor_id: Entity ID of the vendor (metadata filter key).
        context:   The adverse finding explanation used as the retrieval query.
        k:         Number of clause chunks to retrieve.
    """
    store = _get_contract_store()
    if store is None:
        return []
    try:
        docs = store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k, "filter": {"vendor_id": vendor_id}},
        ).invoke(context)
        return [doc.page_content.strip() for doc in docs if doc.page_content.strip()]
    except Exception as exc:
        logger.warning("Contract clause retrieval failed for '%s' (%s)", vendor_id, exc)
        return []


def search_contracts(query: str, vendor_id: str | None = None, k: int = 5) -> list[dict]:
    """Return k relevant contract chunks as dicts with 'text', 'source', 'vendor_id'.

    If vendor_id is given, results are filtered to that vendor's contract only.
    Returns [] when no contracts are indexed.
    """
    store = _get_contract_store()
    if store is None:
        return []
    try:
        kwargs: dict = {"k": k}
        if vendor_id:
            kwargs["filter"] = {"vendor_id": vendor_id}
        docs = store.as_retriever(search_type="similarity", search_kwargs=kwargs).invoke(query)
        return [
            {
                "text":      d.page_content.strip(),
                "source":    d.metadata.get("source", ""),
                "vendor_id": d.metadata.get("vendor_id", ""),
            }
            for d in docs if d.page_content.strip()
        ]
    except Exception as exc:
        logger.warning("Contract search failed (%s)", exc)
        return []


def index_contract(documents: list[Document], vendor_id: str) -> None:
    """Add contract chunks to the contracts Chroma collection.

    Additive — safe to call multiple times. Called exclusively by doc_ingestor.py.

    Args:
        documents: Chunked Documents, each with vendor_id in metadata.
        vendor_id: Entity ID of the vendor (informational; already in metadata).
    """
    from langchain_chroma import Chroma
    global _contract_store
    store = Chroma(
        collection_name=CONTRACTS_COLLECTION,
        embedding_function=_get_embeddings(),
        persist_directory=CHROMA_DIR,
    )
    store.add_documents(documents)
    _contract_store = store
    logger.info("Indexed %d contract chunks for vendor '%s'", len(documents), vendor_id)
