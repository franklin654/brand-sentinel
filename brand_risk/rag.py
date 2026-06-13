"""LangChain-backed RAG store for brand-risk intelligence.

Three persistent ChromaDB collections in the same .chroma/ directory:
  - "incidents"  : historical brand-crisis seed corpus (built by scripts/build_index.py)
  - "contracts"  : per-vendor supplier contracts (rag_contracts.py)
  - "playbook"   : crisis response templates (rag_playbook.py)

The shared HuggingFaceEmbeddings singleton (BAAI/bge-small-en-v1.5) lives here;
rag_contracts and rag_playbook import _get_embeddings from this module so the model
is loaded exactly once per process.

All public functions return [] / "" and degrade gracefully when the index is absent so
smoke.py and offline demo mode are completely unaffected.

Re-exports from rag_contracts and rag_playbook are at the bottom of this file so
all existing import sites (agents.py, doc_ingestor.py, app.py) stay unchanged.
"""
from __future__ import annotations

import logging
import os

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CHROMA_DIR:  str = os.getenv("CHROMA_DIR", ".chroma")
COLLECTION:  str = "incidents"
MODEL_NAME:  str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# ── Module-level singletons (lazy-init) ───────────────────────────────────────
_embeddings = None  # HuggingFaceEmbeddings — shared with rag_contracts + rag_playbook
_store      = None  # Chroma incidents store


# ── Device detection ──────────────────────────────────────────────────────────
def _detect_device() -> str:
    """Return the best available compute device for sentence-transformers.

    Checks CUDA first (covers standard GPU and most ROCm setups), then checks
    torch.version.hip explicitly for AMD ROCm environments where cuda.is_available()
    returns False. Falls back to CPU when neither is found.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.version, "hip", None) is not None:
            logger.info("AMD ROCm detected via torch.version.hip — using cuda device")
            return "cuda"
    except ImportError:
        pass
    return "cpu"


# ── Embeddings singleton ──────────────────────────────────────────────────────
def _get_embeddings():
    """Return the shared HuggingFaceEmbeddings instance, initialising once."""
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        device = _detect_device()
        logger.debug("Initialising HuggingFaceEmbeddings on device: %s", device)
        _embeddings = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


# ── Incidents collection ──────────────────────────────────────────────────────
def _get_store():
    """Lazy-load the incidents Chroma collection. Returns None if absent."""
    global _store
    if _store is not None:
        return _store
    if not os.path.exists(CHROMA_DIR):
        return None
    try:
        from langchain_chroma import Chroma
        _store = Chroma(
            collection_name=COLLECTION,
            embedding_function=_get_embeddings(),
            persist_directory=CHROMA_DIR,
        )
        count = _store._collection.count()
        if count == 0:
            logger.warning("Chroma '%s' is empty — run scripts/build_index.py", COLLECTION)
            _store = None
            return None
        logger.info("Chroma loaded: %d incidents in '%s'", count, COLLECTION)
        return _store
    except Exception as exc:
        logger.warning("Could not load incidents store (%s); incident RAG disabled", exc)
        return None


def retrieve_similar_incidents(narrative: str, k: int = 3) -> list[str]:
    """Return k human-readable strings for the most similar past brand crises.

    Each string: "<entity> (<crisis_type>) → score <n>/100, stock <±n>%: <resolution>"
    Returns [] when the index is absent — callers build the prompt without anchors.

    Args:
        narrative: The social narrative cluster text from TrendSignal.
        k: Number of similar incidents to retrieve.
    """
    store = _get_store()
    if store is None:
        return []
    try:
        docs = store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        ).invoke(narrative)
        return [_format_anchor(doc) for doc in docs]
    except Exception as exc:
        logger.warning("Incident RAG retrieval failed (%s)", exc)
        return []


def index_documents(documents: list[Document]) -> None:
    """Embed and store documents in the incidents collection.

    Overwrites any existing collection so re-runs of build_index.py are idempotent.
    Called exclusively by scripts/build_index.py.

    Args:
        documents: Documents whose page_content will be embedded.
    """
    from langchain_chroma import Chroma
    global _store
    _store = Chroma.from_documents(
        documents=documents,
        embedding=_get_embeddings(),
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
    )
    logger.info("Indexed %d documents into '%s/%s'", len(documents), CHROMA_DIR, COLLECTION)


def _format_anchor(doc: Document) -> str:
    """Convert a retrieved incident Document into a one-line calibration string."""
    m = doc.metadata
    stock = m.get("stock_impact_pct", 0)
    sign  = "+" if stock >= 0 else ""
    return (
        f"{m.get('entity', '?')} ({m.get('crisis_type', '?')}) "
        f"→ score {m.get('risk_score', '?')}/100, "
        f"stock {sign}{stock}%: "
        f"{m.get('resolution', 'no resolution data')}"
    )


# ── Re-exports (backward-compat) ─────────────────────────────────────────────
# Contracts and playbook logic lives in rag_contracts.py / rag_playbook.py.
# Importing from .rag still works for all existing call sites.
from .rag_contracts import (  # noqa: E402
    _get_contract_store, index_contract, retrieve_contract_clauses,
)
from .rag_playbook import (  # noqa: E402
    _get_playbook_store, index_playbook, retrieve_response_template,
)
