"""Persist RSS feed URL configuration so it survives Streamlit restarts.

File: rss_config.json in the project root  (gitignored)
Format: JSON dict  { entity_id -> [url, ...] }
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_RSS_PATH = Path(__file__).parent.parent / "rss_config.json"


def save_rss(config: dict) -> None:
    """Write the RSS feed URL config to disk."""
    _RSS_PATH.write_text(json.dumps(config, indent=2))
    logger.info("RSS config persisted to %s", _RSS_PATH)


def load_rss() -> dict:
    """Return persisted RSS config, or {} if the file is absent or malformed."""
    if not _RSS_PATH.exists():
        return {}
    try:
        return json.loads(_RSS_PATH.read_text())
    except Exception as exc:
        logger.warning("Failed to load RSS config (%s) — using empty config", exc)
        return {}
