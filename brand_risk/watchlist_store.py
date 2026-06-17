"""Persist the active watchlist to JSON so it survives Streamlit restarts.

File: brand_risk/watchlist_override.json  (gitignored)
Format: JSON array of Entity model dicts.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .schemas import Entity

logger = logging.getLogger(__name__)

_OVERRIDE_PATH = Path(__file__).parent.parent / "watchlist_override.json"


def save_watchlist(entities: list[Entity]) -> None:
    """Write entities to the override file."""
    _OVERRIDE_PATH.write_text(
        json.dumps([e.model_dump() for e in entities], indent=2)
    )
    logger.info("Watchlist persisted: %d entities → %s", len(entities), _OVERRIDE_PATH)


def load_watchlist() -> list[Entity] | None:
    """Return persisted watchlist, or None if no override file exists."""
    if not _OVERRIDE_PATH.exists():
        return None
    try:
        raw = json.loads(_OVERRIDE_PATH.read_text())
        return [Entity.model_validate(item) for item in raw]
    except Exception as exc:
        logger.warning("Failed to load watchlist override (%s) — using defaults", exc)
        return None


def clear_watchlist() -> None:
    """Delete the override file, reverting to synthetic demo data."""
    if _OVERRIDE_PATH.exists():
        _OVERRIDE_PATH.unlink()
        logger.info("Watchlist override cleared.")
