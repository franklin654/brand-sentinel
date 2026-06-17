"""Shared helper to load brand_risk/alert_config.json.

Centralises the config path so scheduler.py and notifier.py read from the same
location without duplicating the lookup. agents.py keeps its own _load_alert_config()
inline for backward-compat with Phase 4.
"""
from __future__ import annotations

import json
from pathlib import Path

CONFIG_PATH: Path = Path(__file__).parent.parent / "alert_config.json"


def load_alert_config() -> dict:
    """Return the alert config dict, or {} if the file is absent or malformed."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(CONFIG_PATH.read_text())
    except Exception:
        return {}
