"""Background scheduler — runs the LangGraph pipeline at a fixed interval.

Uses APScheduler's BackgroundScheduler (thread-based, no event loop required).
Results are persisted to SQLite by _synthesise() (Phase 3). Alerts fire via
notifier.notify() when a dossier's risk_score exceeds its configured ceiling.

Public API:
    start(interval_minutes) -> BackgroundScheduler   # call from scripts/run_scheduler.py
"""
from __future__ import annotations

import logging
import os

from apscheduler.schedulers.background import BackgroundScheduler

logger = logging.getLogger(__name__)

SCHEDULE_INTERVAL_MINUTES: int = int(os.getenv("MONITOR_INTERVAL", "15"))


def _run_pipeline_job() -> None:
    """One complete pipeline cycle — log-and-continue on any error."""
    try:
        from . import synthetic_data as data
        from .alert_config_loader import load_alert_config
        from . import notifier
        from .orchestrator import build_graph

        graph = build_graph()
        result = graph.invoke({
            "watchlist": data.WATCHLIST,
            "graph":     data.vendor_graph(),
            "posts":     data.social_stream(),
        })

        alert_cfg = load_alert_config()
        for dossier in result.get("dossiers", []):
            ecfg     = alert_cfg.get(dossier.entity_id, {})
            ceiling  = ecfg.get("risk_score_ceiling", 100)
            channels = ecfg.get("notify", [])
            if dossier.adverse.risk_score >= ceiling and channels:
                notifier.notify(dossier, channels)

        logger.info(
            "Pipeline job complete — %d dossier(s) written to SQLite",
            len(result.get("dossiers", [])),
        )
    except Exception as exc:
        logger.error("Pipeline job failed: %s", exc, exc_info=True)


def start(interval_minutes: int = SCHEDULE_INTERVAL_MINUTES) -> BackgroundScheduler:
    """Create, configure, and start the background scheduler.

    Args:
        interval_minutes: How often to run the pipeline. Reads from MONITOR_INTERVAL
                          env var, defaulting to 15 minutes.

    Returns:
        The running BackgroundScheduler instance. Caller is responsible for calling
        sched.shutdown() on process exit.
    """
    sched = BackgroundScheduler()
    sched.add_job(
        _run_pipeline_job,
        trigger="interval",
        minutes=interval_minutes,
        id="brand_risk_monitor",
        replace_existing=True,
    )
    sched.start()
    logger.info("Scheduler started — pipeline runs every %d minute(s)", interval_minutes)
    return sched
