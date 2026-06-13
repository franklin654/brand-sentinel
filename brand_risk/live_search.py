"""SerpAPI Google News wrapper.

Returns article dicts with keys: title, url, snippet — same shape as
NEWS_CORPUS so adverse_agent requires no changes.

Set SERP_API_KEY in the environment to enable; raises RuntimeError if absent.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

MAX_RESULTS: int = 10
_API_KEY: str = os.getenv("SERP_API_KEY", "")


def fetch_news(entity_name: str, aliases: list[str]) -> list[dict]:
    """Query SerpAPI Google News for entity_name combined with aliases.

    Args:
        entity_name: Primary name of the monitored entity.
        aliases: Alternative names; only the first two are included in the
            query to stay within URL length limits.

    Returns:
        List of article dicts with keys: title, url, snippet.

    Raises:
        RuntimeError: If SERP_API_KEY is not set.
        Exception: Propagates SerpAPI client errors to the caller for logging.
    """
    if not _API_KEY:
        raise RuntimeError("SERP_API_KEY not set — cannot fetch live news")

    from serpapi import GoogleSearch  # google-search-results package

    query = " OR ".join([entity_name] + list(aliases[:2]))
    params = {
        "engine": "google_news",
        "q": query,
        "num": MAX_RESULTS,
        "api_key": _API_KEY,
    }
    logger.debug("SerpAPI query: %s", query)
    results = GoogleSearch(params).get_dict()

    articles: list[dict] = []
    for item in results.get("news_results", [])[:MAX_RESULTS]:
        source = item.get("source", {})
        articles.append({
            "title":     item.get("title", ""),
            "url":       item.get("link", ""),
            "snippet":   item.get("snippet") or source.get("name", ""),
            "published": item.get("date", ""),  # bonus field — shown in UI if present
        })

    logger.info("SerpAPI returned %d articles for '%s'", len(articles), entity_name)
    return articles
