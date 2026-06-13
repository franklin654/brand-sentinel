"""SQLite persistence layer for dossier run history.

DB file: brand_risk.db in the project root (or BRAND_RISK_DB env var).
Schema: single 'runs' table — one row per entity per pipeline run.

Public API:
    save_run(dossiers)      → persist scores from a completed pipeline run
    get_history(entity_id)  → chronological list of run rows for trend charts
    get_all_entities()      → list of (entity_id, entity_name) with history
"""
from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("BRAND_RISK_DB", "brand_risk.db")

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_ts        TEXT    NOT NULL,
    entity_id     TEXT    NOT NULL,
    entity_name   TEXT    NOT NULL,
    risk_score    INTEGER NOT NULL,
    risk_category TEXT    NOT NULL,
    overall_risk  TEXT    NOT NULL,
    social_pct    REAL    DEFAULT 0.0,
    media_pct     REAL    DEFAULT 0.0,
    vendor_pct    REAL    DEFAULT 0.0
);
"""


def _conn() -> sqlite3.Connection:
    """Open (and initialise schema if needed) the SQLite DB."""
    con = sqlite3.connect(DB_PATH)
    con.execute(_CREATE_SQL)
    con.commit()
    return con


def save_run(dossiers: list) -> None:
    """Persist one row per dossier into the runs table."""
    ts = datetime.now(timezone.utc).isoformat()
    rows = [
        (
            ts,
            d.entity_id,
            d.entity_name,
            d.adverse.risk_score,
            d.adverse.risk_category,
            d.overall_risk,
            d.risk_attribution.get("social_pct", 0.0),
            d.risk_attribution.get("media_pct",  0.0),
            d.risk_attribution.get("vendor_pct", 0.0),
        )
        for d in dossiers
    ]
    with _conn() as con:
        con.executemany(
            "INSERT INTO runs "
            "(run_ts, entity_id, entity_name, risk_score, risk_category, "
            "overall_risk, social_pct, media_pct, vendor_pct) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    logger.info("Persisted %d dossier(s) to %s", len(rows), DB_PATH)


def get_history(entity_id: str) -> list[dict]:
    """Return all run rows for an entity ordered chronologically."""
    with _conn() as con:
        cur = con.execute(
            "SELECT run_ts, risk_score, risk_category, social_pct, media_pct, vendor_pct "
            "FROM runs WHERE entity_id = ? ORDER BY run_ts ASC",
            (entity_id,),
        )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_latest(entity_id: str) -> dict | None:
    """Return the most recent run row for an entity, or None if no history."""
    with _conn() as con:
        cur = con.execute(
            "SELECT * FROM runs WHERE entity_id=? ORDER BY run_ts DESC LIMIT 1",
            (entity_id,),
        )
        cols = [c[0] for c in cur.description]
        row = cur.fetchone()
        return dict(zip(cols, row)) if row else None


def get_all_entities() -> list[tuple[str, str]]:
    """Return distinct (entity_id, entity_name) pairs that have history."""
    with _conn() as con:
        cur = con.execute(
            "SELECT DISTINCT entity_id, entity_name FROM runs ORDER BY entity_name"
        )
        return cur.fetchall()
