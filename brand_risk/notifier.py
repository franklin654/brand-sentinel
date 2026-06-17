"""Alert delivery — in-app messages, Slack webhook, and SMTP email.

In-app alerts (collect_alerts) always work — no env vars needed.
Slack and email fire only when the relevant env vars are configured.

Env vars for Slack:   SLACK_WEBHOOK_URL
Env vars for email:   SMTP_HOST, SMTP_PORT (default 587), SMTP_USER, SMTP_PASS,
                      ALERT_EMAIL_TO

Public API:
    collect_alerts(dossiers, alert_cfg) -> list[str]   # for Streamlit toast/banner
    notify(dossier, channels: list[str]) -> None       # external delivery
"""
from __future__ import annotations

import json
import logging
import os
import smtplib
import urllib.request
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

SLACK_WEBHOOK: str = os.getenv("SLACK_WEBHOOK_URL", "")
SMTP_HOST:     str = os.getenv("SMTP_HOST", "")
SMTP_PORT:     int = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER:     str = os.getenv("SMTP_USER", "")
SMTP_PASS:     str = os.getenv("SMTP_PASS", "")
ALERT_EMAIL_TO: str = os.getenv("ALERT_EMAIL_TO", "")


def collect_alerts(dossiers: list, alert_cfg: dict) -> list[str]:
    """Return alert messages for dossiers whose risk_score exceeds ceiling.

    Called in the dashboard after every pipeline run. Results are stored in
    st.session_state.alerts and rendered as st.toast() + persistent banners.
    Works with no external configuration.

    Args:
        dossiers:   List of ReputationDossier objects from the pipeline run.
        alert_cfg:  Dict loaded from alert_config.json (may be empty).
    """
    alerts = []
    for d in dossiers:
        ceiling = alert_cfg.get(d.entity_id, {}).get("risk_score_ceiling", 100)
        if d.adverse.risk_score >= ceiling:
            alerts.append(
                f"**{d.entity_name}** score {d.adverse.risk_score}/100 "
                f"exceeds ceiling {ceiling} — {d.overall_risk.upper()}"
            )
    return alerts


def send_slack(msg: str, webhook_url: str) -> None:
    """POST a plain text alert to a specific Slack incoming webhook URL.

    Args:
        msg:         Alert message text.
        webhook_url: Slack incoming webhook URL from alert_config.json.
    """
    if not webhook_url:
        return
    payload = json.dumps({"text": msg}).encode()
    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=5)
        logger.info("Slack alert delivered to configured webhook.")
    except Exception as exc:
        logger.warning("Slack delivery failed: %s", exc)


def send_email(msg: str, to: str) -> None:
    """Send an alert email to an explicit recipient address.

    Uses SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASS env vars.
    Silent no-op if SMTP_HOST is not configured.

    Args:
        msg: Alert message text.
        to:  Recipient email address from alert_config.json.
    """
    if not to or not SMTP_HOST:
        return
    email_msg = MIMEText(msg)
    email_msg["Subject"] = "Brand Risk Alert"
    email_msg["From"]    = SMTP_USER or "brand-risk@noreply.local"
    email_msg["To"]      = to
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            if SMTP_USER and SMTP_PASS:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASS)
            server.send_message(email_msg)
        logger.info("Email alert delivered to %s.", to)
    except Exception as exc:
        logger.warning("Email delivery failed to %s: %s", to, exc)


def notify(dossier, channels: list[str]) -> None:
    """Deliver an alert for a single dossier via the specified channels.

    Args:
        dossier:  A ReputationDossier whose score breached its ceiling.
        channels: List of channel names — "slack" and/or "email".
    """
    if "slack" in channels:
        _post_slack(dossier)
    if "email" in channels:
        _send_email(dossier)


def _post_slack(dossier) -> None:
    """POST a Slack message via incoming webhook. Silent no-op when URL not set."""
    if not SLACK_WEBHOOK:
        logger.warning("SLACK_WEBHOOK_URL not set — skipping Slack notification")
        return
    payload = json.dumps({
        "text": (
            f"*Brand Risk Alert* — {dossier.entity_name}\n"
            f"Risk score: {dossier.adverse.risk_score}/100 "
            f"({dossier.overall_risk.upper()})\n"
            f"{dossier.headline}"
        )
    }).encode()
    req = urllib.request.Request(
        SLACK_WEBHOOK,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=5)
        logger.info("Slack alert sent for %s", dossier.entity_name)
    except Exception as exc:
        logger.warning("Slack notification failed for %s: %s", dossier.entity_name, exc)


def _send_email(dossier) -> None:
    """Send an alert email via SMTP. Silent no-op when SMTP_HOST / ALERT_EMAIL_TO not set."""
    if not SMTP_HOST or not ALERT_EMAIL_TO:
        logger.warning("SMTP_HOST / ALERT_EMAIL_TO not set — skipping email notification")
        return
    body = (
        f"Entity: {dossier.entity_name}\n"
        f"Risk score: {dossier.adverse.risk_score}/100 — {dossier.overall_risk.upper()}\n"
        f"Headline: {dossier.headline}\n\n"
        f"Explanation: {dossier.adverse.explanation}"
    )
    msg = MIMEText(body)
    msg["Subject"] = (
        f"Brand Risk Alert: {dossier.entity_name} ({dossier.overall_risk.upper()})"
    )
    msg["From"] = SMTP_USER or "brand-risk@noreply.local"
    msg["To"]   = ALERT_EMAIL_TO
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            if SMTP_USER and SMTP_PASS:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        logger.info("Email alert sent for %s", dossier.entity_name)
    except Exception as exc:
        logger.warning("Email notification failed for %s: %s", dossier.entity_name, exc)
