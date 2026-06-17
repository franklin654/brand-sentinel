"""News search helper and empty default corpus.

The Boeing demo scenario has been removed. WATCHLIST, NEWS_CORPUS, social_stream()
and vendor_graph() are all empty — the app starts with no data and the user uploads
their own watchlist, vendor edges, and social posts via the Documents page.

search_news() still works in three tiers:
  1. SerpAPI Google News (when SERP_API_KEY env var is set)
  2. Semantic similarity over NEWS_CORPUS (when sentence-transformers available)
  3. Substring match over NEWS_CORPUS (always available — keeps smoke.py working)
"""
from __future__ import annotations

import logging
import os

import networkx as nx

from .schemas import Entity, SocialPost

logger = logging.getLogger(__name__)

WATCHLIST: list[Entity] = []
PEER_WATCHLIST: list[Entity] = []
NEWS_CORPUS: list[dict] = []

_SIM_THRESHOLD = 0.25


def social_stream(n_per_entity: int = 12) -> list[SocialPost]:
    """Return an empty list — posts must be uploaded via the Documents page."""
    return []


def vendor_graph() -> nx.Graph:
    """Return an empty graph — edges must be uploaded via the Documents page."""
    return nx.Graph()


def search_news(entity_name: str, aliases: list[str]) -> list[dict]:
    """Return relevant news articles for an entity.

    Three-tier resolution:
      1. SerpAPI Google News — when SERP_API_KEY env var is set.
      2. Semantic similarity over NEWS_CORPUS — when sentence-transformers available.
      3. Substring match over NEWS_CORPUS — always available (smoke.py fallback).
    """
    if os.getenv("SERP_API_KEY"):
        try:
            from .live_search import fetch_news
            return fetch_news(entity_name, list(aliases))
        except Exception as exc:
            logger.warning("SerpAPI fetch failed (%s); falling back to corpus", exc)

    try:
        from .embeddings import cosine_scores
        query = " ".join([entity_name] + list(aliases))
        candidates = [f"{a['title']} {a['snippet']}" for a in NEWS_CORPUS]
        scores = cosine_scores(query, candidates)
        return [art for art, score in zip(NEWS_CORPUS, scores) if score >= _SIM_THRESHOLD]
    except ImportError:
        pass

    names = [entity_name.lower(), *[a.lower() for a in aliases]]
    return [
        art for art in NEWS_CORPUS
        if any(nm in (art["title"] + " " + art["snippet"]).lower() for nm in names)
    ]
