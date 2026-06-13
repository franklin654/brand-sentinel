"""Playbook Chroma collection — crisis response template retrieval.

Indexes uploaded PDF/DOCX playbooks (via doc_ingestor.ingest_playbook) into a
single "playbook" collection. Retrieval is entity-agnostic: the LLM gets the
most semantically relevant response template for the crisis narrative at hand.

Public API:
    retrieve_response_template(query, k) -> str   # "" when no playbook indexed
    index_playbook(documents)            -> None
"""
from __future__ import annotations

import logging

from langchain_core.documents import Document

from .rag import CHROMA_DIR, _get_embeddings

logger = logging.getLogger(__name__)

PLAYBOOK_COLLECTION: str = "playbook"

_playbook_store = None  # lazy singleton


def _get_playbook_store():
    """Lazy-load the playbook Chroma collection. Returns None if absent or empty."""
    global _playbook_store
    if _playbook_store is not None:
        return _playbook_store
    import os
    if not os.path.exists(CHROMA_DIR):
        return None
    try:
        from langchain_chroma import Chroma
        store = Chroma(
            collection_name=PLAYBOOK_COLLECTION,
            embedding_function=_get_embeddings(),
            persist_directory=CHROMA_DIR,
        )
        if store._collection.count() == 0:
            return None
        _playbook_store = store
        logger.info("Playbook store loaded: %d chunks", store._collection.count())
        return _playbook_store
    except Exception as exc:
        logger.warning("Could not load playbook store (%s); playbook RAG disabled", exc)
        return None


def retrieve_response_template(query: str, k: int = 1) -> str:
    """Return the top-matching crisis response template for the given narrative query.

    Returns the first relevant chunk as a string, or "" when no playbook is indexed.
    Called by orchestrator._synthesise() to populate ReputationDossier.suggested_response.

    Args:
        query: Combined narrative + explanation text for similarity search.
        k:     Number of chunks to retrieve (only the first is returned as the template).
    """
    store = _get_playbook_store()
    if store is None:
        return ""
    try:
        docs = store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        ).invoke(query)
        return docs[0].page_content.strip() if docs else ""
    except Exception as exc:
        logger.warning("Playbook retrieval failed (%s)", exc)
        return ""


def index_playbook(documents: list[Document]) -> None:
    """Add playbook chunks to the 'playbook' Chroma collection (additive).

    Called exclusively by brand_risk/doc_ingestor.ingest_playbook().

    Args:
        documents: Chunked Documents from the uploaded playbook PDF/DOCX.
    """
    from langchain_chroma import Chroma
    global _playbook_store
    store = Chroma(
        collection_name=PLAYBOOK_COLLECTION,
        embedding_function=_get_embeddings(),
        persist_directory=CHROMA_DIR,
    )
    store.add_documents(documents)
    _playbook_store = store
    logger.info("Indexed %d playbook chunks into '%s'", len(documents), PLAYBOOK_COLLECTION)
