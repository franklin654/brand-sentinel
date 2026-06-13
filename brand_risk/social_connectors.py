"""Live social feed connectors — adapts LangChain loaders to list[SocialPost].

RSS (feedparser via RSSFeedLoader) works without any credentials.
Reddit (PRAW via RedditPostsLoader) is optional — returns [] when env vars absent.

Public API:
    fetch_rss_posts(entity, feed_urls)           -> list[SocialPost]
    fetch_reddit_posts(entity, subreddits)       -> list[SocialPost]
    fetch_all(watchlist, rss_config)             -> list[SocialPost]
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from .schemas import Entity, SocialPost

logger = logging.getLogger(__name__)


def fetch_rss_posts(entity: Entity, feed_urls: list[str]) -> list[SocialPost]:
    """Fetch RSS feed items mentioning the entity (by name or aliases).

    Uses LangChain RSSFeedLoader (requires feedparser>=6.0.12).
    Each matching article becomes a SocialPost whose text is the article summary.

    Args:
        entity:     Entity to match articles against.
        feed_urls:  List of RSS feed URLs to load.
    """
    from langchain_community.document_loaders import RSSFeedLoader

    loader = RSSFeedLoader(urls=feed_urls)
    docs = loader.load()
    terms = [entity.name.lower()] + [a.lower() for a in entity.aliases]
    relevant = [
        d for d in docs
        if any(term in d.page_content.lower() for term in terms)
    ]
    logger.info(
        "RSS: %d/%d articles matched for %s", len(relevant), len(docs), entity.name
    )
    return [_doc_to_post(doc, entity.entity_id) for doc in relevant]


def fetch_reddit_posts(entity: Entity, subreddits: list[str]) -> list[SocialPost]:
    """Fetch Reddit posts mentioning the entity via PRAW.

    Returns [] silently when REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET are not set.
    Requires praw>=7.8.2 to be installed.

    Args:
        entity:     Entity to search for.
        subreddits: List of subreddit names to search within.
    """
    import os

    client_id     = os.getenv("REDDIT_CLIENT_ID", "")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        logger.debug("Reddit creds not set — skipping Reddit connector for %s", entity.name)
        return []

    try:
        from langchain_community.document_loaders import RedditPostsLoader
    except ImportError:
        logger.warning("praw not installed — Reddit connector unavailable")
        return []

    try:
        loader = RedditPostsLoader(
            client_id=client_id,
            client_secret=client_secret,
            user_agent="brand-risk/1.0",
            search_queries=[entity.name] + entity.aliases,
            mode="subreddit",
            subreddits=subreddits,
            number_posts=50,
        )
        return [_doc_to_post(doc, entity.entity_id) for doc in loader.load()]
    except Exception as exc:
        logger.warning("Reddit fetch failed for %s: %s", entity.name, exc)
        return []


def fetch_all(
    watchlist: list[Entity],
    rss_config: dict[str, list[str]],
) -> list[SocialPost]:
    """Fetch RSS posts for all entities that have configured feed URLs.

    Args:
        watchlist:   Full entity list to iterate.
        rss_config:  Mapping of entity_id → list of feed URLs.
    """
    posts: list[SocialPost] = []
    for entity in watchlist:
        urls = rss_config.get(entity.entity_id, [])
        if not urls:
            continue
        try:
            posts.extend(fetch_rss_posts(entity, urls))
        except Exception as exc:
            logger.warning("RSS fetch failed for %s: %s", entity.name, exc)
    return posts


def _doc_to_post(doc, entity_id: str) -> SocialPost:
    """Convert a LangChain Document to a SocialPost schema object."""
    raw_ts = doc.metadata.get("published", "")
    try:
        ts = datetime.fromisoformat(raw_ts).isoformat() if raw_ts else _now()
    except ValueError:
        ts = _now()
    return SocialPost(
        post_id=str(abs(hash(doc.page_content)))[:12],
        entity_id=entity_id,
        text=doc.page_content[:500],
        timestamp=ts,
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
