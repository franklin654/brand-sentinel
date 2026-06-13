#!/usr/bin/env python
"""Headless scheduler entry point — runs the brand-risk pipeline on a cron interval.

Usage:
    python scripts/run_scheduler.py               # default 15-minute interval
    python scripts/run_scheduler.py --interval 1  # 1-minute interval (for testing)

Results are written to brand_risk.db (SQLite). The Streamlit dashboard reads from the
same DB and auto-refreshes when new data is available. Alerts fire to Slack/email for
entities whose risk_score exceeds the ceiling configured in brand_risk/alert_config.json.

Environment variables:
    MONITOR_INTERVAL        Override default interval (minutes)
    SLACK_WEBHOOK_URL       Slack incoming webhook URL
    SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASS / ALERT_EMAIL_TO
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from brand_risk.scheduler import start

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-30s %(levelname)s %(message)s",
)

parser = argparse.ArgumentParser(
    description="Brand-risk headless scheduler",
)
parser.add_argument(
    "--interval",
    type=int,
    default=15,
    metavar="N",
    help="Minutes between pipeline runs (default: 15)",
)
args = parser.parse_args()

sched = start(interval_minutes=args.interval)
print(
    f"Brand-risk scheduler running — pipeline fires every {args.interval} minute(s).\n"
    f"Results → brand_risk.db  ·  Ctrl-C to stop."
)

try:
    while True:
        time.sleep(60)
except (KeyboardInterrupt, SystemExit):
    sched.shutdown(wait=False)
    print("\nScheduler stopped.")
